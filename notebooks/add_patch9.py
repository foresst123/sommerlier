import json
import copy

filepath = "/home/lamkd2/Documents/fullduplex-project/sommelier/notebooks/sommerlier_kaggle.ipynb"
with open(filepath, "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Create new cells for Qwen3 environment setup
qwen3_setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- QWEN3-ASR ISOLATED ENVIRONMENT SETUP ---\n",
        "# We create a separate environment for Qwen3-ASR to avoid huggingface-hub conflicts with WhisperX.\n",
        "QWEN3_ENV_DIR = os.path.join(BASE_DIR, 'qwen3_env')\n",
        "\n",
        "!uv venv {QWEN3_ENV_DIR} --python 3.12\n",
        "\n",
        "# Install PyTorch CUDA 12.6\n",
        "!uv pip install --python {QWEN3_ENV_DIR} torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --extra-index-url https://download.pytorch.org/whl/cu126\n",
        "\n",
        "# Install Qwen3 specific dependencies (Transformers 5.13 requires newer huggingface-hub)\n",
        "!uv pip install --python {QWEN3_ENV_DIR} \"transformers>=5.13.0\" \"huggingface-hub>=1.5.0\" accelerate soundfile librosa\n",
        "\n",
        "# Verify installation\n",
        "!{QWEN3_ENV_DIR}/bin/python -c \"from transformers import AutoProcessor, AutoModelForMultimodalLM; print('✅ Qwen3 environment OK')\"\n"
    ]
}

qwen3_worker_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Write the qwen3_worker.py script to the pipeline directory\n",
        "worker_code = \"\"\"#!/usr/bin/env python3\n",
        "import sys, json, os, torch\n",
        "import soundfile as sf\n",
        "\n",
        "def load_model():\n",
        "    from transformers import AutoProcessor, AutoModelForMultimodalLM\n",
        "    model_name = 'Qwen/Qwen3-ASR-1.7B-hf'\n",
        "    device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES remaps physical GPU 1 to cuda:0\n",
        "    print(json.dumps({'status': 'loading', 'model': model_name}), flush=True)\n",
        "    processor = AutoProcessor.from_pretrained(model_name)\n",
        "    model = AutoModelForMultimodalLM.from_pretrained(model_name, device_map={'': device}, torch_dtype=torch.float16)\n",
        "    model.eval()\n",
        "    print(json.dumps({'status': 'ready', 'device': str(device)}), flush=True)\n",
        "    return model, processor, device\n",
        "\n",
        "def transcribe(model, processor, device, audio_path, language='vi'):\n",
        "    try:\n",
        "        audio_data, sr = sf.read(audio_path, dtype='float32')\n",
        "        if sr != 16000:\n",
        "            import librosa\n",
        "            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)\n",
        "        conversation = [{'role': 'user', 'content': [{'type': 'audio', 'audio_url': 'dummy'}, {'type': 'text', 'text': 'Transcribe the audio in Vietnamese.'}]}]\n",
        "        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)\n",
        "        inputs = processor(text=text, audios=audio_data, return_tensors='pt', sampling_rate=16000).to(device)\n",
        "        with torch.no_grad():\n",
        "            gen_ids = model.generate(**inputs, max_new_tokens=256)\n",
        "            gen_ids = gen_ids[:, inputs.input_ids.size(1):]\n",
        "            response = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]\n",
        "        return response.strip()\n",
        "    except Exception as e:\n",
        "        return f'[ERROR] {e}'\n",
        "\n",
        "def main():\n",
        "    model, processor, device = load_model()\n",
        "    for line in sys.stdin:\n",
        "        line = line.strip()\n",
        "        if not line: continue\n",
        "        try:\n",
        "            request = json.loads(line)\n",
        "        except json.JSONDecodeError:\n",
        "            print(json.dumps({'error': 'invalid JSON'}), flush=True)\n",
        "            continue\n",
        "        cmd = request.get('cmd', 'transcribe')\n",
        "        if cmd == 'quit':\n",
        "            print(json.dumps({'status': 'shutdown'}), flush=True)\n",
        "            break\n",
        "        elif cmd == 'ping':\n",
        "            print(json.dumps({'status': 'ok'}), flush=True)\n",
        "            continue\n",
        "        audio_path = request.get('audio_path', '')\n",
        "        if not audio_path or not os.path.exists(audio_path):\n",
        "            print(json.dumps({'error': f'audio file not found: {audio_path}'}), flush=True)\n",
        "            continue\n",
        "        text = transcribe(model, processor, device, audio_path, request.get('language', 'vi'))\n",
        "        print(json.dumps({'text': text}), flush=True)\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    main()\n",
        "\"\"\"\n",
        "worker_path = os.path.join(PROJECT_DIR, 'podcast-pipeline', 'qwen3_worker.py')\n",
        "with open(worker_path, 'w', encoding='utf-8') as f:\n",
        "    f.write(worker_code)\n",
        "print(f\"✅ Written qwen3_worker.py to {worker_path}\")\n"
    ]
}

# Find where to insert these cells (after environment installation)
insert_idx = -1
for idx, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if any("requirements_proposed.txt" in line for line in source) or any("uv pip install" in line for line in source):
            insert_idx = idx + 1
            break

if insert_idx != -1:
    notebook["cells"].insert(insert_idx, qwen3_setup_cell)
    notebook["cells"].insert(insert_idx + 1, qwen3_worker_cell)

# Now add Patch 9 to modify main_original_ASR_MoE.py
for cell in notebook["cells"]:
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if any("Dynamic patch applied" in line for line in source):
            patch_9 = [
                "    \n",
                "# Patch 9: Refactor Qwen3-ASR to use subprocess worker\n",
                "if os.path.exists(main_script):\n",
                "    with open(main_script, 'r', encoding='utf-8') as f:\n",
                "        code = f.read()\n",
                "    \n",
                "    # 9a: Replace load model\n",
                "    old_load = '            from transformers import AutoProcessor, AutoModelForMultimodalLM\\n            logger.debug(\" * Loading Qwen3-ASR (VN, slot 3)\")\\n            canary_model = AutoModelForMultimodalLM.from_pretrained(\\n                \"Qwen/Qwen3-ASR-1.7B-hf\", \\n                device_map={\"*\":  device_2}, \\n                torch_dtype=torch.float16\\n            )\\n            canary_model.processor = AutoProcessor.from_pretrained(\"Qwen/Qwen3-ASR-1.7B-hf\")\\n            logger.debug(f\" * PhoWhisper + Qwen3-ASR loaded successfully\")'\n",
                "    old_load_2 = '            from transformers import AutoProcessor, AutoModelForMultimodalLM\\n            logger.debug(\" * Loading Qwen3-ASR (VN, slot 3)\")\\n            canary_model = AutoModelForMultimodalLM.from_pretrained(\\n                \"Qwen/Qwen3-ASR-1.7B-hf\", \\n                device_map=\"auto\", \\n                torch_dtype=torch.float16\\n            )\\n            canary_model.processor = AutoProcessor.from_pretrained(\"Qwen/Qwen3-ASR-1.7B-hf\")\\n            logger.debug(f\" * PhoWhisper + Qwen3-ASR loaded successfully\")'\n",
                "    new_load = '''            import subprocess as _sp, json, tempfile\\n            qwen3_env_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), \"..\", \"qwen3_env\", \"bin\", \"python\")\\n            qwen3_worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), \"qwen3_worker.py\")\\n            qwen3_env_for_worker = os.environ.copy()\\n            qwen3_env_for_worker[\"CUDA_VISIBLE_DEVICES\"] = \"1\"\\n            logger.debug(f\" * Starting Qwen3 worker: {qwen3_env_bin}\")\\n            qwen3_process = _sp.Popen([qwen3_env_bin, qwen3_worker_script], stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True, bufsize=1, env=qwen3_env_for_worker)\\n            canary_model = None\\n            _qwen3_ready = False\\n            for _line in iter(qwen3_process.stdout.readline, \"\"):\\n                _line = _line.strip()\\n                if not _line: continue\\n                try:\\n                    _msg = json.loads(_line)\\n                    if _msg.get(\"status\") == \"ready\":\\n                        _qwen3_ready = True\\n                        break\\n                except:\\n                    pass\\n            if _qwen3_ready:\\n                logger.debug(\" * Qwen3 worker loaded successfully\")\\n            else:\\n                logger.error(\" * Qwen3 worker failed to start\")'''\n",
                "    \n",
                "    import re\n",
                "    # regex replace since it might be slightly different\n",
                "    code = re.sub(r'            from transformers import AutoProcessor, AutoModelForMultimodalLM.*?loaded successfully\"\\)', new_load, code, flags=re.DOTALL)\n",
                "    \n",
                "    # 9b: Replace inference\n",
                "    old_inf = r'        # Run Qwen3-ASR for inference\\n.*?            return response.strip\\(\\)'\n",
                "    new_inf = '''        # Run Qwen3-ASR via worker\\n        try:\\n            if not _qwen3_ready or qwen3_process is None:\\n                return \"\"\\n            with tempfile.NamedTemporaryFile(suffix=\".wav\", delete=False) as tmp_audio:\\n                tmp_path = tmp_audio.name\\n            import soundfile as sf\\n            sf.write(tmp_path, segment_audio_16k, 16000)\\n            qwen3_process.stdin.write(json.dumps({\"cmd\": \"transcribe\", \"audio_path\": tmp_path}) + \"\\\\n\")\\n            resp_line = qwen3_process.stdout.readline()\\n            if os.path.exists(tmp_path):\\n                os.remove(tmp_path)\\n            if not resp_line: return \"\"\\n            return json.loads(resp_line).get(\"text\", \"\")'''\n",
                "    code = re.sub(old_inf, new_inf, code, flags=re.DOTALL)\n",
                "    \n",
                "    with open(main_script, 'w', encoding='utf-8') as f:\n",
                "        f.write(code)\n",
                "    print(f\"✅ Patch 9: Subprocess Qwen3 worker applied to {main_script}\")\n"
            ]
            source.extend(patch_9)
            cell["source"] = source
            break

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print("Done")
