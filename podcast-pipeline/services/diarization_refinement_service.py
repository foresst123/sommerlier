from typing import List
from schemas.transcript import TranscriptSegment
import torch

class DiarizationRefinementService:
    """Uses a local LLM (Qwen) to refine speaker labels and text based on dialogue context."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.model = None
        self.tokenizer = None
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
            
    def refine(self, segments: List[TranscriptSegment], prompt: str) -> List[TranscriptSegment]:
        """Call local Qwen LLM to fix hallucination and text errors."""
        self._load_model()
        if not self.model:
            return segments
            
        if self.logger: self.logger.info("Running LLM Refinement on all segments...")
        
        from tqdm import tqdm
        for seg in tqdm(segments, desc="[LLM] Đang tinh chỉnh câu"):
            w_text = seg.text_whisper or ""
            p_text = seg.text_phowhisper or ""
            q_text = seg.text_qwen3 or ""
            
            # Skip if all are empty
            if not w_text and not p_text and not q_text:
                continue
                
            prompt = (
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
                "C1 (Người đang nói) — Phải duy trì ĐẠI TỪ NHÂN XƯNG (anh/tôi/chị/em/mình) thống nhất. "
                "Cùng một Speaker không thể xưng 'tôi' ở câu này và 'bạn' ở câu ngay sau.\n"
                "C2 (Thời gian) — Nếu thời gian ngắn (ví dụ: < 1s), đừng cố ghép thành một câu dài "
                "hoàn chỉnh, ưu tiên giữ nguyên từ đệm/từ ngắn tương xứng với thời lượng.\n"
                "C3 (Trạng thái hội thoại) — Nếu là 'CÓ (Đang tranh lời/Nói đè)': Âm thanh rất ồn "
                "và ASR thường xuyên nghe sai lệch. Bạn được phép BẢO THỦ HƠN, ưu tiên chọn lọc "
                "những từ chung nhất giữa 3 bản, loại bỏ các từ vô nghĩa hoặc tạp âm.\n\n"

                "### ĐỊNH DẠNG ĐẦU RA (bắt buộc tuân thủ nghiêm ngặt)\n"
                "- Chỉ xuất ra transcript tiếng Việt cuối cùng, không kèm gì khác.\n"
                "- Không giải thích lý do, không liệt kê phương án, không viết 'có thể là', "
                "'không chắc chắn'.\n"
                "- Không đặt trong dấu ngoặc kép, không dùng Markdown, không tiêu đề.\n"
                "- Không thêm tiền tố như 'Kết quả:', 'Transcript:', 'Output:'.\n"
            )
            
            is_overlap = "CÓ (Đang tranh lời/Nói đè)" if getattr(seg, 'tse', False) else "KHÔNG (Độc thoại)"
            
            user_msg = (
                f"Thông tin Ngữ cảnh:\n"
                f"- Người đang nói: {seg.speaker}\n"
                f"- Thời gian: {seg.start:.2f}s - {seg.end:.2f}s\n"
                f"- Trạng thái hội thoại: {is_overlap}\n"
                f"---\n"
                f"Bản dịch 1 (Whisper): {w_text}\n"
                f"Bản dịch 2 (PhoWhisper): {p_text}\n"
                f"Bản dịch 3 (Qwen3-ASR): {q_text}\n\n"
                f"Dựa vào luật và ngữ cảnh, câu hoàn chỉnh là:"
            )
            
            conversation = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg}
            ]
            
            try:
                text_input = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.0,
                        do_sample=False
                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    refined_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    
                # Ghi đè câu đã được sửa lỗi mượt mà vào trường text chính
                seg.text = refined_text
            except Exception as e:
                if self.logger: self.logger.warning(f"LLM failed on segment {seg.index}: {e}")

        return segments
