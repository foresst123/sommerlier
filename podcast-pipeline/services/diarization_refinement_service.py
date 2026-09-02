import os
from typing import List
from difflib import SequenceMatcher
from algorithms.asr.hallucination import diacritic_ratio, foreign_script_ratio
from algorithms.asr.rover import normalize_token
from schemas.transcript import TranscriptSegment
import torch
import re
# Hoisted out of the per-segment loop: this is ~1200 constant tokens that would
# otherwise be rebuilt and re-tokenized for every single segment.
# Below this word-level similarity to every input transcript, the refinement is
# discarded. 0.5 keeps genuine cleanups (punctuation, a corrected word) while
# rejecting output that came from somewhere else.
ACCEPT_SIMILARITY = 0.5
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

FUSION_SYSTEM_PROMPT = (
    "Bạn hợp nhất 3 bản transcript ASR tiếng Việt của CÙNG một đoạn audio ngắn thành 1 bản duy nhất. Bạn không nghe được audio, chỉ có 3 bản text.\n"
    "\n"
    "Bản 1 = Whisper. Bản 2 = PhoWhisper. Bản 3 = Qwen3.\n"
    "Ba model này sai theo những kiểu KHÁC NHAU và biết trước được. Dùng hiểu biết đó để chọn.\n"
    "\n"
    # The single largest correctable failure: 47 of 303 wrong segments (15.5%)
    # were the three transcripts concatenated rather than one chosen.
    # Stating the rule abstractly was not enough; the contrasting examples are.
    "### LUẬT TUYỆT ĐỐI\n"
    "Đây là BA CÁCH NGHE KHÁC NHAU của cùng một câu nói, KHÔNG PHẢI ba câu nói nối tiếp nhau. Kết quả dài xấp xỉ MỘT bản, không phải tổng của ba bản.\n"
    "- SAI: 'Bản 1: dạ / Bản 2: nhờ / Bản 3: yeah' → 'dạ nhờ yeah'\n"
    "- ĐÚNG: chọn một → 'dạ'\n"
    "- SAI: 'xong mình thấy vậy' + 'sau mình thấy vậy' → ghép cả hai\n"
    "- ĐÚNG: 'xong mình thấy vậy'\n"
    "Khi các bản chỉ khác nhau ở một vài từ, chúng là CÙNG một câu: chọn từ, đừng cộng câu.\n"
    "\n"
    "Mỗi từ trong kết quả PHẢI xuất hiện trong ít nhất 1 trong 3 bản. Bạn CHỌN giữa các bản, không VIẾT LẠI. Khi phân vân, chọn phương án ít thay đổi nhất.\n"
    "\n"
    # Each model fails in its own measurable way across the 932-segment
    # evaluation, and naming those ways is what lets the fusion pick between
    # them rather than average them. The claims below are all measured, not
    # guessed -- see tools/asr_evaluation/.
    "### ĐẶC TÍNH TỪNG BẢN\n"
    "\n"
    "Bản 1 (Whisper)\n"
    "- Đáng tin khi: đoạn chỉ có một từ đệm đứng một mình; đoạn rất ngắn 1-2 từ.\n"
    "- Hay sai khi: câu dài (>11 từ) — kém nhất; nuốt mất từ (cái, ờ, mà, là); đổi số thành chữ số ('hai'→'2'); nhầm đại từ ('anh'→'em'); bỏ mất phần lặp; viết 'yeah' thành 'dạ'/'vâng'.\n"
    "- Nếu Bản 1 khác cả hai bản kia, nó chỉ đúng ~7% — theo hai bản kia.\n"
    "\n"
    "Bản 2 (PhoWhisper)\n"
    "- Đáng tin khi: câu có từ đệm ở ĐẦU hoặc GIỮA (ờ, ừm, ờm, à) — bản duy nhất giữ được; câu ngắn 2-6 từ; giữ phần lặp.\n"
    "- Hay sai khi: câu có từ tiếng Anh — TỆ NHẤT, luôn phiên âm thành tiếng Việt vô nghĩa; câu dài; chèn thêm từ không có ('cái', 'là', cả từ đệm 'ờ' giả); nhầm đại từ vùng miền ('cô'→'cổ', 'tôi'→'tui'); sinh token rác 'unk'; Việt hóa tên riêng ('youtube'→'túp').\n"
    "\n"
    "Bản 3 (Qwen3)\n"
    "- Đáng tin khi: câu dài (≥11 từ) — hơn hẳn, càng dài càng hơn; câu có số liệu; câu có đại từ nhân xưng; tên riêng; giữ phần lặp; từ 'yeah' (bản duy nhất viết đúng).\n"
    "- Hay sai khi: đoạn cực ngắn 1-2 từ — hay TRẢ VỀ RỖNG (43 lần, gần như toàn bộ là 'ừ/ờ/ừm'); xóa từ đệm cho câu 'sạch'; chèn thêm 'cái', 'là', 'nó', 'mà' để câu mượt hơn; DỊCH NGUYÊN CÂU SANG TIẾNG ANH khi gặp nhiều từ tiếng Anh; dịch từ chức năng ('thì'→'then').\n"
    "\n"
    "### CÁCH CHỌN\n"
    "1. Chỗ cả 3 bản giống nhau: giữ nguyên.\n"
    "2. Chỗ khác nhau: chọn phương án hợp lý nhất trong mạch câu. Không đếm phiếu máy móc — 1 bản đúng vẫn thắng 2 bản sai — nhưng khi không có lý do rõ ràng để chọn khác, phương án được 2 bản đồng ý thường đúng (khoảng 7/10 lần).\n"
    "3. Bản trống hoặc chứa 'unk' là bản đó NGHE HỤT, không phải người nói im lặng. Bỏ qua nó, chọn giữa hai bản còn lại. Đừng bao giờ trả về rỗng chỉ vì một bản rỗng.\n"
    "4. Câu dài, nhiều mệnh đề: lấy bản mạch lạc nhất làm khung (thường là Bản 3), rồi sửa từng từ lẻ theo hai bản kia nếu chúng đồng ý với nhau.\n"
    "5. Đoạn rất ngắn: chọn bản CÓ CHỮ. Đừng chọn bản rỗng, đừng ghép hai bản ngắn lại.\n"
    "6. Tên riêng, số liệu: chọn bản rõ nghĩa nhất, giữ nguyên dạng chữ ('hai' vẫn là 'hai', không đổi thành '2'). Không tự sửa theo hiểu biết của bạn.\n"
    "7. Câu dở dang trong cả 3 bản: giữ dở dang. Không viết tiếp cho trọn ý.\n"
    "8. Giữ nguyên thứ tự từ của bản bạn chọn. Không đảo vế câu cho xuôi tai.\n"
    "\n"
    # Filler behaviour splits by position, and the split is large: on fillers
    # inside a sentence PhoWhisper keeps 16/18 against 2/18 for the others,
    # while on a filler standing alone Whisper leads Qwen3 52% to 8%.
    "### TỪ ĐỆM (ờ, ừ, ừm, ờm, à, dạ, vâng)\n"
    "Từ đệm là một phần của lời nói, phải giữ.\n"
    "- Nếu MỘT bản mở đầu bằng từ đệm mà các bản kia không có: GIỮ từ đệm đó. Bản bắt được nó thường đúng.\n"
    "- Nếu cả đoạn chỉ là một từ đệm: chọn bản có từ đệm tiếng Việt. Bản rỗng và bản dịch sang tiếng Anh ('is', 'yes', 'el') đều sai.\n"
    "- 'dạ' và 'yeah' là hai từ khác nhau, không thay cho nhau. Bản nào viết 'yeah' thì thường là nghe đúng 'yeah'.\n"
    "- Đừng lược bỏ từ đệm để câu gọn hơn.\n"
    "\n"
    # The previous prompt told the model to collapse every adjacent repeat,
    # which was wrong on the 38 segments where the speaker really did repeat.
    # Agreement separates the two cases: where >=2 transcripts carry the
    # repeat it is genuine (29/38), and where exactly one does it is an ASR
    # artefact (24/24).
    "### TỪ LẶP LIỀN KỀ\n"
    "Người nói có lặp từ thật ('mình mình cũng tự hào', 'từ từ', 'lâng lâng').\n"
    "- Nếu HAI bản trở lên cùng có phần lặp: giữ phần lặp — người nói lặp thật.\n"
    "- Nếu CHỈ MỘT bản có phần lặp còn hai bản kia không: bỏ phần lặp — đó là lỗi ASR.\n"
    "\n"
    # Loanwords are the corpus's hardest class. PhoWhisper scores 4-12% on
    # them at every density -- it transliterates rather than transcribes --
    # while Qwen3 translates 28 segments wholesale into English. The two
    # failures pull in opposite directions, so both need naming.
    "### TỪ TIẾNG ANH XEN TIẾNG VIỆT\n"
    "Người nói trộn thuật ngữ tiếng Anh (mindset, fix, format, unlearn, relearn, fail, chill, top, debate, platform, content, feedback, launch, evolve, impact, brand, test, speaker, profile...).\n"
    "- ASR hay phiên âm chúng thành từ tiếng Việt vô nghĩa: 'mai sét' (mindset), 'phích' (fix), 'mát' (format), 'phèo' (fail), 'chiêu' (chill), 'lên' (unlearn), 'túp' (youtube), 'ninh' (unlearning).\n"
    "- Nếu một bản giữ chữ tiếng Anh còn bản khác cho ra từ tiếng Việt nghe na ná mà vô nghĩa trong câu: CHỌN BẢN TIẾNG ANH. Đây là luật chắc chắn nhất — bản phiên âm gần như luôn sai.\n"
    "- NGƯỢC LẠI, KHÔNG DỊCH tiếng Việt sang tiếng Anh. Nếu một bản biến cả câu thành tiếng Anh ('nhưng sau đó họ fail là bởi vì họ fix' → 'but after that they failed because they fixed'), BỎ bản đó, dùng hai bản kia.\n"
    "- Kết quả luôn là câu tiếng Việt có dấu, chỉ giữ nguyên các từ tiếng Anh mà người nói thực sự dùng.\n"
    "\n"
    "### GIỮ NGUYÊN VĂN NÓI\n"
    "Đây là hội thoại tự nhiên. Giữ từ đệm, giữ đại từ nhân xưng đúng như trong bản gốc, không thay bằng từ trang trọng hơn. Không thêm từ nối ('cái', 'là', 'nó', 'mà') để câu mượt hơn — nếu một bản có từ đó mà hai bản kia không có, nhiều khả năng nó được chèn thêm. Chỉ thêm dấu câu để dễ đọc.\n"
    "\n"
    "### ĐẦU RA\n"
    "Chỉ xuất transcript tiếng Việt của đoạn này, không gì khác.\n"
    "- Không nhận xét, không đánh giá, không giải thích. Nếu bạn nghĩ đoạn này là quảng cáo, vô nghĩa hay bị lỗi, bạn VẪN xuất transcript của nó. Việc lọc bỏ do hệ thống khác làm, không phải việc của bạn.\n"
    "- Không thêm tiền tố ('Kết quả:', 'Transcript:'), không ngoặc kép, không Markdown.\n"
    "- Nếu cả 3 bản đều trống, xuất ra chuỗi rỗng.\n"
    ""
)


# Above this share of segments failing, the stage is treated as broken rather
# than as having hit a few awkward inputs.
REFINE_FAILURE_LIMIT = float(os.environ.get("REFINE_FAILURE_LIMIT", "0.5"))

# Segments this short, or this few words, are left with their ROVER text.
#
# Fusion needs three transcripts that disagree in a way context can settle. A
# backchannel gives the model one or two words and no sentence to place them
# in, so there is nothing to choose between -- and what came back instead was
# invention: "dạ" -> "dạ nhờ yeah", "ừ" -> "ừ từ đúng", "đúng không" -> "bàu
# đùa". Hand-checking 932 segments found the models already right on these and
# refinement wrong, so the cheapest correct move is not to ask.
#
# The bound is deliberately tight. Refinement earns its keep on ordinary
# segments (286 fixed against 38 broken), and every segment skipped here is one
# it no longer gets to fix.
REFINE_MIN_SECONDS = float(os.environ.get("REFINE_MIN_SECONDS", "1.0"))
REFINE_MIN_WORDS = int(os.environ.get("REFINE_MIN_WORDS", "2"))


def too_short_to_refine(seg) -> bool:
    """Whether ``seg`` is a backchannel that refinement should leave alone.

    Duration and word count are both checked because either can be the honest
    signal: a 0.4s "ừ" is short in time, while a slowly drawn-out "dạ" is short
    only in words. The word count uses the longest of the three transcripts --
    the shortest is often blank, and one model dropping a word is not evidence
    the segment is a backchannel.
    """
    words = max((len(t.split()) for t in
                 (seg.text_whisper, seg.text_phowhisper, seg.text_qwen3) if t),
                default=0)
    if words and words <= REFINE_MIN_WORDS:
        return True

    # A missing or reversed span says nothing; only trust a real duration.
    start, end = getattr(seg, "start", None), getattr(seg, "end", None)
    if start is not None and end is not None and end > start:
        return (end - start) < REFINE_MIN_SECONDS
    return False


class DiarizationRefinementService:
    """Uses a local LLM (Qwen) to refine speaker labels and text based on dialogue context."""

    # The ~1200-token system prompt dominates each sequence, so even a modest
    # batch builds a large KV cache; 2 fits alongside the ASR models on a T4.
    def __init__(self, logger=None, batch_size: int = 4, model_name: str = None,
                 torch_dtype: str = "bfloat16", prefix_cache: bool = False):
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.batch_size = batch_size
        self.rejected = 0
        self.torch_dtype = torch_dtype
        # Every request repeats the same ~1200-token system prompt, and without
        # this each batch re-runs the attention over it from scratch. Caching
        # its keys and values once per stage cuts the prefill by 43% at batch 2
        # and 82% at batch 24 -- the saving grows with the batch, which is why
        # it is worth having before moving to a card that can hold a big one.
        # Off by default: it needs headroom for the cache and a model whose
        # generate() accepts a prepared past_key_values, and a T4 running batch
        # 2 has neither to spare.
        self.prefix_cache = prefix_cache
        self._prefix = None          # (input_ids, past_key_values, length)
        self._prefix_key = None      # the system prompt it was built from
        self._prefix_failed = False  # give up quietly after one failure
        # Qwen2.5-3B-Instruct: ~6.2GB in bf16, which leaves room on a 14.5GB T4
        # for whatever else is resident, and it is what produced the refinement
        # results this pipeline has actually been measured on (0 of 36 segments
        # damaged on the last full run).
        #
        # A 9B model was set here previously on the argument that it covers more
        # languages. It does not fit: ~18GB in bf16 against 14.5GB of card, so
        # device_map="auto" would silently spill it to CPU and refinement would
        # crawl rather than fail. Anything larger needs its VRAM checked on the
        # target hardware first, not reasoned about from the model card.
        # SOMMELIER_LLM still overrides without a code change.
        self.model_name = (model_name or os.environ.get(
            "SOMMELIER_LLM", "Qwen/Qwen2.5-3B-Instruct"))
        
    def _load_model(self):
        if self.model is not None:
            return
            
        if self.logger: self.logger.info(f"Loading LLM {self.model_name} for refinement...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # bfloat16 keeps fp32's exponent range, so activations cannot
            # overflow the way they can in fp16. It only runs natively from
            # Ampere onwards -- Turing emulates it -- so the profile picks.
            dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto"
            )
            self.model.eval()
            if self.logger: self.logger.info("LLM loaded successfully.")
        except Exception as e:
            # Do not swallow this. refine() treats a missing model as "nothing
            # to do" and hands the transcripts straight back, so a failed load
            # used to look exactly like a successful no-op run: the stage logged
            # an error, the pipeline carried on, and the output was silently
            # un-refined. Raising here means the run stops at the point the
            # problem is still visible.
            msg = (f"Failed to load refinement LLM {self.model_name}: {e}. "
                   "Set SOMMELIER_LLM to a model that fits this GPU, or drop "
                   "--llm_refinement to skip the stage deliberately.")
            if self.logger:
                self.logger.error(msg)
            raise RuntimeError(msg) from e

    def reset_stats(self):
        """Clear per-file counters; the instance is reused across a batch."""
        self.rejected = 0

    def unload(self):
        """Release the refinement model and its VRAM.

        Nothing freed it before, so a ~6.2GB bf16 model sat in VRAM from the
        first refinement until the process exited. That is invisible when one
        file is processed and then the process ends, and costly when several
        files run back to back and every later stage competes for what this
        stage is no longer using.
        """
        if self.model is None and self.tokenizer is None:
            return
        # The prefix cache holds tensors on the model's device; dropping the
        # model without it would leave that memory pinned for the rest of the run.
        self._release_prefix()
        self._prefix_failed = False
        self.model = None
        self.tokenizer = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if self.logger:
            self.logger.info("Unloaded refinement LLM from VRAM")

    def _accept(self, seg, refined: str) -> bool:
        """Whether the model's output is a fusion of this segment's transcripts.

        A prompt cannot guarantee behaviour, and the failures it lets through
        are the expensive kind: a segment that carries a neighbour's words is a
        mislabelled training example, worse than an unpolished one. Compare the
        result against the three inputs and keep the ROVER text when it does not
        resemble any of them.
        """
        sources = [t for t in (seg.text_whisper, seg.text_phowhisper, seg.text_qwen3)
                   if t and t.strip()]
        if not sources:
            return False

        # A whole-text similarity check cannot see a local insert: one segment
        # came back with a Chinese translation spliced into the middle of an
        # otherwise faithful 145-word Vietnamese sentence and still scored 0.77.
        # Vietnamese is Latin script, so any foreign-script run the inputs did
        # not contain is the model writing rather than choosing.
        if foreign_script_ratio(refined) > foreign_script_ratio(" ".join(sources)) + 0.01:
            if self.logger:
                self.logger.warning(
                    f"[LLM] rejected refinement for segment {seg.index} "
                    f"(introduced non-Latin script); keeping ROVER text"
                )
            return False

        # The check above only sees non-Latin script, and English is Latin, so
        # a segment translated wholesale into English passes it untouched:
        # "nhưng sau đó họ fail là bởi vì họ fix" came back as "but after that
        # they failed because they fixed" and scored 0.00 on both sides.
        #
        # Diacritics are the difference. Compare against the inputs rather than
        # thresholding the output alone -- Vietnamese transcribed without
        # diacritics is legitimate, and only a drop relative to what the ASR
        # models produced is evidence the model translated instead of choosing.
        source_diacritics = diacritic_ratio(" ".join(sources))
        if source_diacritics >= 0.25 and diacritic_ratio(refined) < source_diacritics * 0.4:
            if self.logger:
                self.logger.warning(
                    f"[LLM] rejected refinement for segment {seg.index} "
                    f"(translated out of Vietnamese: diacritics "
                    f"{source_diacritics:.2f} -> {diacritic_ratio(refined):.2f}); "
                    "keeping ROVER text"
                )
            return False

        # Length is checked before similarity, not after it. Fusing three
        # readings of one utterance cannot outrun the longest of them, but a
        # concatenation ("dạ" + "nhờ" + "yeah" -> "dạ nhờ yeah") contains every
        # input word and so scores high on the similarity test below. Ordering
        # the checks this way turns 41 of the corpus's 303 wrong segments into
        # rejections at the cost of one correct segment.
        longest_words = max(len(t.split()) for t in sources)
        if longest_words and len(refined.split()) > longest_words * 1.5:
            if self.logger:
                self.logger.warning(
                    f"[LLM] rejected refinement for segment {seg.index} "
                    f"(concatenated inputs: {len(refined.split())} words vs "
                    f"longest input {longest_words}); keeping ROVER text"
                )
            return False

        # Compare on folded tokens. Punctuation is exactly what refinement is
        # supposed to add, and on a one-word backchannel it is the whole
        # difference: "Ừ?" against "Ừ" scores 0.0 as raw words and would reject
        # every corrected backchannel in the corpus.
        def fold(text):
            return [normalize_token(w) for w in text.split() if normalize_token(w)]

        target = fold(refined)
        if not target:
            return False
        best = max(SequenceMatcher(None, target, fold(t)).ratio() for t in sources)
        if best >= ACCEPT_SIMILARITY:
            return True

        if self.logger:
            self.logger.warning(
                f"[LLM] rejected refinement for segment {seg.index} "
                f"(too dissimilar, sim={best:.2f}); keeping ROVER text"
            )
        return False

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
        # _build_user_message is pure string work over the pre-refinement text,
        # so the segments are independent and this parallelises cleanly. It is
        # the one part of refinement that is CPU-bound: generation itself is on
        # the GPU and stays sequential.
        indices = [i for i, seg in enumerate(segments)
                   if (seg.text_whisper or seg.text_phowhisper or seg.text_qwen3)
                   and not too_short_to_refine(seg)]

        skipped = sum(1 for seg in segments
                      if (seg.text_whisper or seg.text_phowhisper or seg.text_qwen3)
                      and too_short_to_refine(seg))
        if skipped and self.logger:
            self.logger.info(
                f"[LLM] Skipping {skipped} backchannel segment(s) "
                f"(< {REFINE_MIN_SECONDS}s or <= {REFINE_MIN_WORDS} words); "
                "keeping their ROVER text")

        workers = int(os.environ.get("OMP_NUM_THREADS", "1"))
        if len(indices) > 64 and workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                messages = list(ex.map(lambda i: self._build_user_message(segments, i),
                                       indices))
        else:
            # Below that, the pool costs more to start than the work it saves.
            messages = [self._build_user_message(segments, i) for i in indices]

        pending = [(segments[i], m) for i, m in zip(indices, messages)]

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

        tail = f", {self.rejected} rejected as unfaithful" if self.rejected else ""
        if self.logger:
            if failed_count:
                self.logger.warning(
                    f"LLM refinement: {refined_count} segments refined, "
                    f"{failed_count} left unrefined (original ROVER text kept){tail}"
                )
            else:
                self.logger.info(
                    f"LLM refinement: {refined_count} segments refined{tail}")

        # A few failed batches are a bad input; most of them failing is a broken
        # configuration -- almost always the GPU being too full for the model
        # that was asked for. Returning the un-refined text either way made a
        # run that produced nothing look identical to one that worked, and the
        # ledger then recorded the file as done. Raise so the file is marked
        # failed and retried, instead of silently landing in the dataset with a
        # stage missing.
        if pending and failed_count / len(pending) > REFINE_FAILURE_LIMIT:
            raise RuntimeError(
                f"LLM refinement failed on {failed_count} of {len(pending)} "
                f"segments ({100.0 * failed_count / len(pending):.0f}%). The "
                "usual cause is too little free VRAM for "
                f"{self.model_name}: check the out-of-memory warnings above, "
                "lower models.refinement.batch_size, or use a smaller model.")

        return segments

    # ------------------------------------------------------------------
    def _split_prompt(self, system_prompt: str, user_msg: str):
        """Chat template split into its shared prefix and the per-request tail.

        Rendering the system turn alone and the full exchange, then checking the
        first is a prefix of the second, is what makes the split safe across
        chat templates -- their exact control tokens differ per model family,
        so slicing at a hardcoded offset would silently corrupt the prompt.
        Returns None when the template does not lay out that way.
        """
        tok = self.tokenizer
        try:
            prefix = tok.apply_chat_template(
                [{"role": "system", "content": system_prompt}],
                tokenize=False, add_generation_prompt=False)
            full = tok.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return None
        # An empty or near-empty prefix means the template folded the system
        # turn into the user turn: there is nothing shared to cache, and
        # proceeding would build a cache of zero tokens that later reshapes
        # into an error.
        if not prefix or not full.startswith(prefix) or len(prefix) < 16:
            return None
        return prefix, full[len(prefix):]

    def _build_prefix(self, system_prompt: str, sample_user: str) -> bool:
        """Run the shared prefix once and keep its KV cache. False if unusable."""
        if self._prefix_failed or not self.prefix_cache:
            return False
        if self._prefix is not None and self._prefix_key == system_prompt:
            return True

        split = self._split_prompt(system_prompt, sample_user)
        if split is None:
            self._prefix_failed = True
            if self.logger:
                self.logger.info(
                    "Prefix cache off: this chat template does not start with a "
                    "standalone system turn")
            return False

        prefix_text, _ = split
        try:
            ids = self.tokenizer(prefix_text, return_tensors="pt",
                                 add_special_tokens=False).input_ids.to(self.model.device)
            if ids.shape[1] == 0:
                raise ValueError("shared prefix tokenised to nothing")
            with torch.no_grad():
                out = self.model(input_ids=ids, use_cache=True)
            self._prefix = (ids, out.past_key_values, ids.shape[1])
            self._prefix_key = system_prompt
            if self.logger:
                self.logger.info(
                    f"Prefix cache built: {ids.shape[1]} shared tokens will skip "
                    "prefill on every batch")
            return True
        except Exception as e:
            self._prefix_failed = True
            self._prefix = None
            if self.logger:
                self.logger.warning(f"Prefix cache unavailable, using full prompts: {e}")
            return False

    def _expand_prefix(self, batch_size: int):
        """A fresh copy of the cached prefix, widened to the batch.

        Rebuilt per batch rather than mutated: the cache grows as generation
        proceeds, so handing the same object to a second batch would feed it
        the first batch's tokens.

        transformers 5 exposes batch_repeat_interleave for exactly this and
        keeps its internal layout private; 4.x stored key_cache/value_cache as
        plain lists. Both are handled, and an unrecognised layout returns None
        so the caller falls back rather than guessing -- a mis-shaped cache
        yields wrong text, not an exception.
        """
        import copy as _copy

        try:
            src = self._prefix[1]
        except (TypeError, IndexError):
            return None

        # transformers >= 5: official API, cache internals stay private.
        if hasattr(src, "batch_repeat_interleave"):
            try:
                clone = _copy.deepcopy(src)
                clone.batch_repeat_interleave(batch_size)
                return clone
            except Exception:
                return None

        def _widen(t):
            return t.expand(batch_size, *t.shape[1:]).contiguous()

        # transformers 4.x legacy tuple form.
        if isinstance(src, (tuple, list)):
            try:
                return tuple(tuple(_widen(t) for t in layer) for layer in src)
            except Exception:
                return None

        keys = getattr(src, "key_cache", None)
        values = getattr(src, "value_cache", None)
        if keys is not None and values is not None:
            try:
                legacy = tuple((_widen(k), _widen(v)) for k, v in zip(keys, values))
                from transformers.cache_utils import DynamicCache
                if hasattr(DynamicCache, "from_legacy_cache"):
                    return DynamicCache.from_legacy_cache(legacy)
                return legacy
            except Exception:
                return None
        return None

    def _release_prefix(self):
        self._prefix = None
        self._prefix_key = None

    def _prepare_cached_inputs(self, batch, system_prompt, inputs, gen_kwargs) -> bool:
        """Swap `inputs` to tail-only tokens backed by the cached prefix.

        Mutates `inputs` and `gen_kwargs` in place and returns whether it
        succeeded. On any doubt it changes nothing and returns False, leaving
        the caller on the full-prompt path: a wrong cache produces wrong
        Vietnamese rather than an exception, so the bar for using it is that
        every step verified cleanly.
        """
        tok = self.tokenizer
        try:
            tails = []
            for _, user_msg in batch:
                split = self._split_prompt(system_prompt, user_msg)
                if split is None:
                    return False
                tails.append(split[1])

            enc = tok(tails, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(self.model.device)

            past = self._expand_prefix(len(batch))
            if past is None:
                # An unrecognised cache class: stop trying rather than risk it.
                self._prefix_failed = True
                self._release_prefix()
                if self.logger:
                    self.logger.info(
                        "Prefix cache off: this transformers version returns a "
                        "cache type that cannot be safely expanded")
                return False

            prefix_len = self._prefix[2]
            mask = torch.cat([
                torch.ones(len(batch), prefix_len, dtype=enc.attention_mask.dtype,
                           device=enc.attention_mask.device),
                enc.attention_mask,
            ], dim=1)

            inputs["input_ids"] = enc.input_ids
            inputs["attention_mask"] = mask
            gen_kwargs["past_key_values"] = past
            # generate() slices the prompt off by input_ids width, and the
            # caller measures the same way, so both stay consistent.
            return True
        except Exception as e:
            self._prefix_failed = True
            self._release_prefix()
            if self.logger:
                self.logger.warning(
                    f"Prefix cache disabled after an error, continuing with full "
                    f"prompts: {e}")
            return False

    def _refine_batch(self, batch, system_prompt):
        """Refine one batch in place. Returns (succeeded, refined_count)."""
        tokenizer = self.tokenizer
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
                 enable_thinking=False,
            )
            for _, user_msg in batch
        ]

        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
            gen_kwargs = dict(
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Reuse the shared prefix's KV cache when one is available. The
            # tails are re-tokenised on their own and the attention mask is
            # widened to cover the cached span, so the model sees exactly the
            # same sequence -- only the prefix's attention is not recomputed.
            # Anything unexpected falls through to the full-prompt path below,
            # which is the one that has always run.
            past = None
            if self._build_prefix(system_prompt, batch[0][1]):
                past = self._prepare_cached_inputs(batch, system_prompt, inputs, gen_kwargs)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs.input_ids.shape[1]
            decoded = tokenizer.batch_decode(
                generated_ids[:, prompt_len:], skip_special_tokens=True
            )
            count = 0
            for (seg, _), refined_text in zip(batch, decoded):
                refined_text = _THINK_RE.sub("", refined_text)
                refined_text = refined_text.split("</think>")[-1].strip()
                if not refined_text:
                    continue
                if not self._accept(seg, refined_text):
                    self.rejected += 1
                    continue
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
