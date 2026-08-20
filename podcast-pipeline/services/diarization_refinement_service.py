from typing import List
from schemas.transcript import TranscriptSegment
import torch

# Hoisted out of the per-segment loop: this is ~1200 constant tokens that would
# otherwise be rebuilt and re-tokenized for every single segment.
FUSION_SYSTEM_PROMPT = (
    "Bạn hợp nhất 3 bản transcript ASR tiếng Việt của CÙNG một đoạn audio thành "
    "1 bản duy nhất. Bạn không nghe được audio, chỉ có 3 bản text.\n\n"

    "### NGUYÊN TẮC GỐC\n"
    "Mỗi từ trong kết quả PHẢI xuất hiện trong ít nhất 1 trong 3 bản. "
    "Bạn CHỌN giữa các bản, không VIẾT LẠI. Khi phân vân, chọn phương án "
    "ít thay đổi nhất.\n\n"

    "### CÁCH CHỌN\n"
    "1. Chỗ cả 3 bản giống nhau: giữ nguyên.\n"
    "2. Chỗ khác nhau: chọn bản nghe hợp lý nhất trong mạch câu. Không đếm "
    "phiếu máy móc — 1 bản đúng vẫn thắng 2 bản sai.\n"
    "3. Tên riêng, số liệu: chọn bản rõ nghĩa nhất. Không tự sửa theo hiểu "
    "biết của bạn.\n"
    "4. Câu dở dang trong cả 3 bản: giữ dở dang. Không viết tiếp cho trọn ý.\n"
    "5. Từ bị lặp liền kề do lỗi ASR (ví dụ 'gì hết. gì hết.'): giữ lại một lần.\n\n"

    "### GIỮ NGUYÊN VĂN NÓI\n"
    "Đây là hội thoại tự nhiên. Giữ từ đệm (ừ, à, ờ, thì, mà, kiểu như), giữ "
    "đại từ nhân xưng đúng như trong bản gốc, không thay bằng từ trang trọng "
    "hơn. Chỉ thêm dấu câu để dễ đọc.\n\n"

    "### ĐẦU RA\n"
    "Chỉ xuất transcript tiếng Việt của đoạn này, không gì khác.\n"
    "- Không nhận xét, không đánh giá, không giải thích. Nếu bạn nghĩ đoạn này "
    "là quảng cáo, vô nghĩa hay bị lỗi, bạn VẪN xuất transcript của nó. "
    "Việc lọc bỏ do hệ thống khác làm, không phải việc của bạn.\n"
    "- Không thêm tiền tố ('Kết quả:', 'Transcript:'), không ngoặc kép, "
    "không Markdown.\n"
    "- Nếu cả 3 bản đều trống, xuất ra chuỗi rỗng.\n"
)


class DiarizationRefinementService:
    """Uses a local LLM (Qwen) to refine speaker labels and text based on dialogue context."""

    # The ~1200-token system prompt dominates each sequence, so even a modest
    # batch builds a large KV cache; 2 fits alongside the ASR models on a T4.
    def __init__(self, logger=None, batch_size: int = 4):
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size
        # Qwen2.5-3B is extremely fast, smart enough for this task, and only takes ~6GB VRAM
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        
    def _load_model(self):
        if self.model is not None:
            return
            
        if self.logger: self.logger.info(f"Loading LLM {self.model_name} for refinement...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model.eval()
            if self.logger: self.logger.info("LLM loaded successfully.")
        except Exception as e:
            if self.logger: self.logger.error(f"Failed to load LLM: {e}")
            
    def _build_user_message(self, segments, i) -> str:
        """Build the fusion request for one segment.

        The previous segment's text is deliberately absent. It was supplied as
        pronoun context, guarded by a "TUYỆT ĐỐI KHÔNG copy" warning, and the
        model copied it anyway: 5 of 8 damaged segments in the last run carried
        a neighbouring segment's words, including one where speaker 1's speech
        was emitted under speaker 2's label. Text the model cannot see is text
        it cannot borrow.
        """
        seg = segments[i]
        duration = seg.end - seg.start

        # Duration is the one contextual fact worth stating: it tells the model
        # whether a long transcript is plausible at all. A 0.24s clip cannot
        # hold a sentence, and saying so is cheaper than a rule about it.
        note = ""
        if duration < 1.0:
            note = (f"\nĐoạn này chỉ dài {duration:.2f} giây — nhiều nhất là "
                    f"một vài từ. Bản dịch nào dài hơn thế là ASR bịa, bỏ qua.\n")
        elif getattr(seg, 'tse', False):
            note = ("\nĐoạn này có hai người nói chồng lên nhau, ASR dễ nghe sai. "
                    "Ưu tiên phần cả 3 bản đồng ý.\n")

        return (
            f"Bản 1: {seg.text_whisper or ''}\n"
            f"Bản 2: {seg.text_phowhisper or ''}\n"
            f"Bản 3: {seg.text_qwen3 or ''}\n"
            f"{note}"
            f"\nTranscript hợp nhất:"
        )

    def refine(self, segments: List[TranscriptSegment], prompt: str = None) -> List[TranscriptSegment]:
        """Call local Qwen LLM to fix hallucination and text errors.

        ``prompt`` overrides the built-in fusion system prompt when supplied.
        """
        self._load_model()
        if not self.model:
            return segments

        if self.logger: self.logger.info("Running LLM Refinement on all segments...")

        system_prompt = prompt or FUSION_SYSTEM_PROMPT

        # The previous segment's text feeds the next one's context, so build all
        # messages up front against the pre-refinement text. Batching keeps the
        # GPU busy instead of running one 1200-token prefill at a time.
        pending = []
        for i, seg in enumerate(segments):
            if not (seg.text_whisper or seg.text_phowhisper or seg.text_qwen3):
                continue
            pending.append((seg, self._build_user_message(segments, i)))

        if not pending:
            return segments

        tokenizer = self.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Decoder-only models need left padding for correct batched generation.
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        from tqdm import tqdm
        refined_count = 0
        failed_count = 0
        try:
            with tqdm(total=len(pending), desc="[LLM] Đang tinh chỉnh câu") as bar:
                start = 0
                while start < len(pending):
                    size = min(self.batch_size, len(pending) - start)
                    batch = pending[start:start + size]
                    ok, count = self._refine_batch(batch, system_prompt)

                    # A batch that does not fit is halved rather than dropped:
                    # the whole refinement stage silently no-ops otherwise.
                    while not ok and len(batch) > 1:
                        size = max(1, len(batch) // 2)
                        if self.logger:
                            self.logger.info(f"Retrying LLM refinement with batch size {size}")
                        batch = pending[start:start + size]
                        ok, count = self._refine_batch(batch, system_prompt)

                    if ok:
                        refined_count += count
                    else:
                        failed_count += len(batch)

                    start += len(batch)
                    bar.update(len(batch))
        finally:
            tokenizer.padding_side = original_padding_side

        if self.logger:
            if failed_count:
                self.logger.warning(
                    f"LLM refinement: {refined_count} segments refined, "
                    f"{failed_count} left unrefined (original ROVER text kept)"
                )
            else:
                self.logger.info(f"LLM refinement: {refined_count} segments refined")

        return segments

    def _refine_batch(self, batch, system_prompt):
        """Refine one batch in place. Returns (succeeded, refined_count)."""
        tokenizer = self.tokenizer
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True
            )
            for _, user_msg in batch
        ]

        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=None,  
                    top_p=None,        
                    top_k=None,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_len = inputs.input_ids.shape[1]
            decoded = tokenizer.batch_decode(
                generated_ids[:, prompt_len:], skip_special_tokens=True
            )
            count = 0
            for (seg, _), refined_text in zip(batch, decoded):
                refined_text = refined_text.strip()
                if refined_text:
                    seg.text = refined_text
                    count += 1
            return True, count

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            if self.logger:
                indices = ", ".join(seg.index for seg, _ in batch)
                self.logger.warning(f"LLM out of memory on [{indices}] (batch {len(batch)}): {e}")
            return False, 0
        except Exception as e:
            if self.logger:
                indices = ", ".join(seg.index for seg, _ in batch)
                self.logger.warning(f"LLM failed on segments [{indices}]: {e}")
            return False, 0
