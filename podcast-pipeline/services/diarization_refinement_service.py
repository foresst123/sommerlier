from typing import List
from schemas.transcript import TranscriptSegment
import torch

# Hoisted out of the per-segment loop: this is ~1200 constant tokens that would
# otherwise be rebuilt and re-tokenized for every single segment.
FUSION_SYSTEM_PROMPT = (
    "### VAI TRÒ\n"
    "Bạn là hệ thống hợp nhất (fusion) 3 bản transcript ASR tiếng Việt của CÙNG một đoạn "
    "audio hội thoại/podcast, thành 1 bản transcript duy nhất chính xác nhất.\n"
    "Bạn KHÔNG nghe được audio. Mọi quyết định chỉ được dựa trên bằng chứng có trong "
    "3 bản transcript đầu vào, không dùng kiến thức ngoài.\n\n"

    "### THỨ TỰ ƯU TIÊN (khi các mục tiêu xung đột, ưu tiên mục có số nhỏ hơn)\n"
    "1. Trung thực với nội dung có bằng chứng — không bịa, không suy diễn.\n"
    "2. Không hallucination (đặc biệt: outro, quảng cáo, câu lặp vô nghĩa).\n"
    "3. Đúng từ, tên riêng, số liệu, thuật ngữ.\n"
    "4. Giữ đúng ý nghĩa gốc của câu nói.\n"
    "5. Giữ khẩu ngữ tự nhiên (không văn phong hóa).\n"
    "6. Thêm dấu câu để dễ đọc (không đổi nghĩa).\n"
    "KHÔNG ưu tiên câu văn hay/mượt nếu điều đó đổi nội dung.\n\n"

    "### QUY TẮC CHỌN GIỮA 3 BẢN (áp dụng cho từng cụm từ/câu khác biệt)\n"
    "R1 — Không mặc định bản xuất hiện nhiều (2/3) là đúng. Đánh giá theo độ hợp lý ngữ "
    "cảnh, không theo số phiếu.\n"
    "R2 — Một từ chỉ có ở 1 bản vẫn được chọn nếu nó khớp ngữ cảnh và nghe hợp lý hơn 2 "
    "bản còn lại.\n"
    "R3 — Không được ghép nửa câu bản A + nửa câu bản B thành câu mới nếu không bản nào "
    "chứa nguyên cụm đó.\n"
    "R4 — Câu chưa trọn nghĩa trong cả 3 bản thì giữ nguyên trạng thái chưa trọn nghĩa, "
    "không tự hoàn thiện cho \"nghe xuôi\".\n"
    "R5 — Khi không đủ căn cứ để quyết định, luôn chọn phương án ít suy diễn nhất "
    "(bảo thủ), không đoán cho đẹp câu.\n\n"

    "### CHỐNG HALLUCINATION (bắt buộc kiểm tra)\n"
    "H1 — Nếu chỉ 1/3 bản có nội dung dạng: mời subscribe, cảm ơn theo dõi, chào tạm biệt, "
    "quảng cáo, kêu gọi hành động không liên quan -> xoá, coi là hallucination.\n"
    "H2 — Nếu 1 bản có nội dung còn 2 bản kia trống/im lặng -> nghi ngờ mạnh là "
    "hallucination, mặc định không giữ trừ khi nội dung đó rất ngắn và hợp lý với mạch câu "
    "ngay trước/sau.\n"
    "H3 — Không thêm tên người, địa danh, công ty, số liệu, sự kiện không xuất hiện trong "
    "bất kỳ bản nào, kể cả khi bạn 'biết' nó đúng ngoài đời thực.\n"
    "H4 — Mỗi từ trong bản output cuối cùng phải truy được về ít nhất 1 trong 3 bản input.\n\n"

    "### TÊN RIÊNG & THUẬT NGỮ\n"
    "- Nếu 1 bản cho tên riêng/thuật ngữ có nghĩa, rõ ràng, còn 2 bản kia cho phiên âm vô "
    "nghĩa -> ưu tiên bản có nghĩa.\n"
    "- Không tự đoán hoặc sửa tên riêng nếu cả 3 bản đều không rõ ràng.\n\n"

    "### SỐ LIỆU (số, ngày, giờ, %, tiền, năm, phiên bản)\n"
    "- Giữ đúng số liệu có bằng chứng rõ nhất trong 3 bản.\n"
    "- Không tự sửa số liệu chỉ vì thấy 'hợp lý hơn'.\n"
    "- Nếu 3 bản mâu thuẫn số liệu và không rõ bản nào đúng -> chọn bản có vẻ ít lỗi ASR "
    "nhất, tuyệt đối không bịa giá trị mới không có trong cả 3 bản.\n\n"

    "### VĂN PHONG\n"
    "- Đây là hội thoại nói tự nhiên, không phải văn viết. Giữ nguyên từ đệm khẩu ngữ "
    "(ừ, à, ờ, ừm, thì, mà, kiểu như...) trừ khi rõ ràng là rác/lỗi ASR.\n"
    "- Không paraphrase, không thay từ khẩu ngữ bằng từ trang trọng hơn.\n"
    "- Chỉ thêm dấu câu (. , ? ...) để dễ đọc, dấu câu không được tạo thêm nghĩa mới.\n\n"

    "### TẬN DỤNG THÔNG TIN NGỮ CẢNH (BẮT BUỘC)\n"
    "C1 (Người đang nói) — Giữ nguyên đại từ nhân xưng đúng như trong 3 bản dịch. "
    "Không đổi 'anh' thành 'tôi' hay ngược lại.\n"
    "C2 (Thời gian) — Nếu thời gian ngắn (ví dụ: < 1s), đừng cố ghép thành một câu dài "
    "hoàn chỉnh, ưu tiên giữ nguyên từ đệm/từ ngắn tương xứng với thời lượng.\n"
    "C3 (Trạng thái hội thoại) — Nếu là 'CÓ (Đang tranh lời/Nói đè)': Âm thanh rất ồn "
    "và ASR thường xuyên nghe sai lệch. Bạn được phép BẢO THỦ HƠN, ưu tiên chọn lọc "
    "những từ chung nhất giữa 3 bản, loại bỏ các từ vô nghĩa hoặc tạp âm.\n"
    "C4 (Lặp từ) — LỖI RẤT PHỔ BIẾN CẦN TRÁNH: Đôi khi các bản dịch ASR bị lặp từ ở cuối câu "
    "(ví dụ: 'gì hết. gì hết.'). BẠN PHẢI TỰ ĐỘNG CẮT BỎ CÁC TỪ BỊ LẶP LẠI VÔ NGHĨA NÀY.\n\n"

    "### ĐỊNH DẠNG ĐẦU RA (bắt buộc tuân thủ nghiêm ngặt)\n"
    "- Chỉ xuất ra transcript tiếng Việt cuối cùng, không kèm gì khác.\n"
    "- TUYỆT ĐỐI KHÔNG nhận xét về đoạn text. Không viết 'Quảng cáo', "
    "'nên cắt bỏ', 'đây là hallucination', 'không có nội dung' hay bất kỳ "
    "đánh giá nào. Nếu bạn cho rằng đoạn này là quảng cáo hay vô nghĩa, "
    "vẫn phải xuất ra transcript của nó, không phải nhận xét của bạn.\n"
    "- Chỉ dùng nội dung có trong 3 bản dịch của CHÍNH đoạn này. Không thêm "
    "câu từ đoạn khác, không tự viết tiếp cho đủ ý.\n"
    "- Không giải thích lý do, không liệt kê phương án, không viết 'có thể là', "
    "'không chắc chắn'.\n"
    "- Không đặt trong dấu ngoặc kép, không dùng Markdown, không tiêu đề.\n"
    "- Không thêm tiền tố như 'Kết quả:', 'Transcript:', 'Output:'.\n"
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
        is_overlap = "CÓ (Đang tranh lời/Nói đè)" if getattr(seg, 'tse', False) else "KHÔNG (Độc thoại)"

        return (
            f"Thông tin Ngữ cảnh:\n"
            f"- Người đang nói: {seg.speaker}\n"
            f"- Thời lượng: {seg.end - seg.start:.2f} giây\n"
            f"- Trạng thái hội thoại: {is_overlap}\n"
            f"---\n"
            f"Bản dịch 1 (Whisper): {seg.text_whisper or ''}\n"
            f"Bản dịch 2 (PhoWhisper): {seg.text_phowhisper or ''}\n"
            f"Bản dịch 3 (Qwen3-ASR): {seg.text_qwen3 or ''}\n\n"
            f"Chỉ xuất ra transcript tiếng Việt của đoạn này. "
            f"Không giải thích, không nhận xét, không thêm nhãn.\n"
            f"Câu hoàn chỉnh:"
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
