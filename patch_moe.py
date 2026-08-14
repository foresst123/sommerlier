import re

file_path = "/home/lamkd2/Documents/fullduplex-project/sommelier/podcast-pipeline/main_original_ASR_MoE.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Pyannote to community-1
content = content.replace(
    '"pyannote/speaker-diarization-3.1"',
    '"pyannote/speaker-diarization-community-1"'
)

# 2. Fix the diarization loop to actually use Pyannote when --dia3 is True
old_diar_loop = """            for chunk in diar_chunks:
                predicted_segments, _ = diar_model.diarize(
                    audio=chunk["path"], batch_size=1, include_tensor_outputs=True
                )
                chunk_df = sortformer_dia(predicted_segments)
                if not chunk_df.empty:"""

new_diar_loop = """            for chunk in diar_chunks:
                if args.dia3:
                    import torchaudio
                    waveform, sr = torchaudio.load(chunk["path"])
                    waveform = waveform.to(device)
                    segments = dia_pipeline({"waveform": waveform, "sample_rate": sr})
                    data = []
                    for turn, _, speaker in segments.itertracks(yield_label=True):
                        data.append({"speaker": speaker, "start": turn.start, "end": turn.end})
                    import pandas as pd
                    chunk_df = pd.DataFrame(data) if data else pd.DataFrame(columns=["speaker", "start", "end"])
                else:
                    predicted_segments, _ = diar_model.diarize(
                        audio=chunk["path"], batch_size=1, include_tensor_outputs=True
                    )
                    chunk_df = sortformer_dia(predicted_segments)
                if not chunk_df.empty:"""

content = content.replace(old_diar_loop, new_diar_loop)

# 3. Replace ChunkFormer initialization with Qwen2-Audio
old_canary_init = """            logger.debug(" * Loading ChunkFormer-CTC (VN, slot 3)")
            canary_model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie")
            logger.debug(f" * PhoWhisper + ChunkFormer loaded on {device}")"""

new_canary_init = """            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
            logger.debug(" * Loading Qwen2-Audio (VN, slot 3)")
            canary_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct", 
                device_map="auto", 
                torch_dtype=torch.float16
            )
            canary_model.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
            logger.debug(f" * PhoWhisper + Qwen2-Audio loaded successfully")"""

content = content.replace(old_canary_init, new_canary_init)

# 4. Replace run_chunkformer_task with Qwen task logic
old_chunkformer_task = """    def run_chunkformer_task(segment_audio_16k):
        # ChunkFormer-CTC expects a wav path; tempfile is opened inside the thread.
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                sf.write(temp_wav.name, segment_audio_16k, 16000)
                temp_wav.flush()
                text = canary_model.endless_decode(
                    audio_path=temp_wav.name,
                    chunk_size=64,
                    left_context_size=128,
                    right_context_size=128,
                    total_batch_duration=1800,
                )
                return text if isinstance(text, str) else str(text)
        except Exception as e:
            logger.error(f"ChunkFormer failed: {e}")
            return \"\""""

new_chunkformer_task = """    def run_chunkformer_task(segment_audio_16k):
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

content = content.replace(old_chunkformer_task, new_chunkformer_task)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied successfully.")
