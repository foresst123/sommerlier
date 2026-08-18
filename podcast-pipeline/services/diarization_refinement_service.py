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
                
            sys_prompt = (
                "Bạn là một chuyên gia biên tập transcript tiếng Việt cho podcast hội thoại tự nhiên, đồng thời là bộ tổng hợp đầu ra ASR (ASR transcript fusion) có độ chính xác cao.\n\n"
                "Bạn sẽ nhận được 3 bản transcript được tạo bởi 3 mô hình ASR khác nhau nhưng đều tương ứng với CÙNG MỘT đoạn âm thanh.\n\n"
                "NHIỆM VỤ:\n"
                "Phân tích và đối chiếu cả 3 bản transcript, sau đó tạo ra DUY NHẤT một câu/đoạn tiếng Việt phản ánh trung thực nhất những gì thực sự được nói trong audio.\n\n"
                "MỤC TIÊU ƯU TIÊN THEO THỨ TỰ:\n"
                "1. Trung thành với lời nói thực tế.\n"
                "2. Không hallucination.\n"
                "3. Đúng từ, đúng tên riêng và thuật ngữ.\n"
                "4. Giữ đúng ý nghĩa và ngữ cảnh hội thoại.\n"
                "5. Tự nhiên, dễ đọc bằng tiếng Việt.\n"
                "6. Không tự ý thêm thông tin không có trong các transcript.\n\n"
                "QUY TẮC QUAN TRỌNG:\n\n"
                "### 1. KHÔNG ĐƯỢC MÙ QUÁNG CHỌN MAJORITY VOTE\n"
                "Không mặc định rằng bản xuất hiện nhiều nhất là đúng. Hãy xác định bản nào phù hợp nhất với ngữ cảnh, tên riêng và thông tin thực tế.\n\n"
                "### 2. CHỐNG HALLUCINATION CỰC MẠNH\n"
                "ĐẶC BIỆT loại bỏ các hallucination kiểu YouTube/podcast như: 'Hãy subscribe kênh', 'Đừng quên đăng ký', 'Cảm ơn các bạn đã theo dõi', v.v.\n"
                "Nếu chỉ một model sinh ra một câu quảng bá nhưng các model khác không có, hãy coi đó là HALLUCINATION và loại bỏ.\n\n"
                "### 3. KHÔNG ĐƯỢC BỊA NỘI DUNG ĐỂ 'LÀM CÂU ĐẸP HƠN'\n"
                "Không thêm từ, diễn giải hay tự sáng tác phần còn thiếu.\n\n"
                "### 4. XỬ LÝ TÊN RIÊNG VÀ THUẬT NGỮ\n"
                "ASR tiếng Việt thường nhận dạng sai tên riêng tiếng Anh. Nếu một transcript có tên riêng hợp lý và các transcript khác là phiên âm sai, ưu tiên tên riêng hợp lý.\n\n"
                "### 5. GIỮ LẠI ĐẶC ĐIỂM CỦA LỜI NÓI KHẨU NGỮ\n"
                "Giữ lại từ đệm tự nhiên ('ừ', 'à', 'ờ', 'kiểu như'). Không biến thành văn phong sách vở cứng nhắc.\n\n"
                "### 6. XỬ LÝ KHOẢNG LẶNG\n"
                "Nếu một model tạo ra nội dung trong khi các model khác để trống, ưu tiên khả năng audio là khoảng lặng và LOẠI BỎ nội dung rác.\n\n"
                "### 7. KHÔNG ĐƯỢC TRẢ VỀ NHIỀU PHƯƠNG ÁN\n"
                "Chỉ xuất ra MỘT kết quả tốt nhất. Không giải thích, không viết 'Có thể là...'\n\n"
                "ĐẦU RA:\n"
                "CHỈ IN RA transcript tiếng Việt cuối cùng.\n"
                "KHÔNG giải thích.\n"
                "KHÔNG đặt trong dấu ngoặc kép.\n"
                "KHÔNG thêm markdown.\n"
                "KHÔNG thêm tiền tố như 'Kết quả:'."
            )
            
            user_msg = f"Bản dịch 1 (Whisper): {w_text}\nBản dịch 2 (PhoWhisper): {p_text}\nBản dịch 3 (Qwen3-ASR): {q_text}\n\nCâu hoàn chỉnh:"
            
            conversation = [
                {"role": "system", "content": sys_prompt},
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
