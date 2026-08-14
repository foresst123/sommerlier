import re

file_path = "/home/lamkd2/Documents/fullduplex-project/sommelier/podcast-pipeline/main_original_ASR_MoE.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace initialization
old_canary_init = """            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
            logger.debug(" * Loading Qwen2-Audio (VN, slot 3)")
            canary_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct", 
                device_map="auto", 
                torch_dtype=torch.float16
            )
            canary_model.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
            logger.debug(f" * PhoWhisper + Qwen2-Audio loaded successfully")"""

new_canary_init = """            from transformers import AutoProcessor, AutoModelForMultimodalLM
            logger.debug(" * Loading Qwen3-ASR (VN, slot 3)")
            canary_model = AutoModelForMultimodalLM.from_pretrained(
                "Qwen/Qwen3-ASR-1.7B-hf", 
                device_map="auto", 
                torch_dtype=torch.float16
            )
            canary_model.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B-hf")
            logger.debug(f" * PhoWhisper + Qwen3-ASR loaded successfully")"""

content = content.replace(old_canary_init, new_canary_init)

# Replace task execution
old_chunkformer_task = """    def run_chunkformer_task(segment_audio_16k):
        # Run Qwen2-Audio for inference
        try:
            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "dummy"},
                    {"type": "text", "text": "Transcribe the audio in Vietnamese."},
                ]}
            ]
            processor = canary_model.processor
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, audios=segment_audio_16k, return_tensors="pt", sampling_rate=16000).to(device)
            
            gen_ids = canary_model.generate(**inputs, max_new_tokens=256)
            gen_ids = gen_ids[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response.strip()
        except Exception as e:
            logger.error(f"Qwen2-Audio failed: {e}")
            return \"\""""

new_chunkformer_task = """    def run_chunkformer_task(segment_audio_16k):
        # Run Qwen3-ASR for inference
        try:
            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "dummy"},
                    {"type": "text", "text": "Transcribe the audio in Vietnamese."},
                ]}
            ]
            processor = canary_model.processor
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, audios=segment_audio_16k, return_tensors="pt", sampling_rate=16000).to(device)
            
            gen_ids = canary_model.generate(**inputs, max_new_tokens=256)
            gen_ids = gen_ids[:, inputs.input_ids.size(1):]
            response = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return response.strip()
        except Exception as e:
            logger.error(f"Qwen3-ASR failed: {e}")
            return \"\""""

content = content.replace(old_chunkformer_task, new_chunkformer_task)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied for Qwen3-ASR.")
