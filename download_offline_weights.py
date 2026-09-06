import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download

def download_models(token=None):
    # Set cache directories
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_weights")
    hf_cache = os.path.join(base_dir, "huggingface")
    torch_cache = os.path.join(base_dir, "torch")
    
    os.makedirs(hf_cache, exist_ok=True)
    os.makedirs(torch_cache, exist_ok=True)
    
    os.environ["HF_HOME"] = hf_cache
    os.environ["TORCH_HOME"] = torch_cache
    
    print(f"=====================================================")
    print(f"BẮT ĐẦU TẢI MODELS VỀ THƯ MỤC: {base_dir}")
    print(f"=====================================================")
    
    models_to_download = [
        "vinai/PhoWhisper-large",
        "Qwen/Qwen3-ASR-1.7B-hf"
    ]
    
    # Models requiring HF Token
    auth_models = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
        "pyannote/wespeaker-voxceleb-resnet34-LM"
    ]

    # 1. Download Public Models
    for repo_id in models_to_download:
        print(f"\n[+] Đang tải mô hình: {repo_id}...")
        try:
            snapshot_download(repo_id=repo_id, cache_dir=hf_cache, local_files_only=False)
            print(f"    -> Xong!")
        except Exception as e:
            print(f"    -> Lỗi: {e}")

    # 2. Download Faster-Whisper Model
    print(f"\n[+] Đang tải mô hình: Whisper large-v3-turbo (faster-whisper)...")
    try:
        from faster_whisper import download_model
        download_model("large-v3-turbo", cache_dir=hf_cache)
        print(f"    -> Xong!")
    except Exception as e:
        print(f"    -> Lỗi: {e}. Đang thử tải qua HuggingFace Hub...")
        try:
            snapshot_download(repo_id="deepdml/faster-whisper-large-v3-turbo-ct2", cache_dir=hf_cache)
            print(f"    -> Xong!")
        except Exception as e2:
            print(f"    -> Cả 2 cách đều lỗi: {e2}")

    # 3. Download Silero VAD
    print(f"\n[+] Đang tải mô hình: Silero VAD (PyTorch Hub)...")
    try:
        import torch
        torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True)
        print(f"    -> Xong!")
    except Exception as e:
        print(f"    -> Lỗi: {e}")

    # 4. Download Authenticated Models
    print(f"\n[+] Đang tải các mô hình cần xác thực (Pyannote)...")
    if token:
        from huggingface_hub import login
        login(token=token)
        for repo_id in auth_models:
            print(f"    Đang tải {repo_id}...")
            try:
                snapshot_download(repo_id=repo_id, cache_dir=hf_cache, token=token)
                print(f"    -> Xong!")
            except Exception as e:
                print(f"    -> Lỗi tải {repo_id}: {e}")
    else:
        print(f"    [BỎ QUA] Bạn chưa cung cấp --token. Pyannote Diarization yêu cầu HuggingFace Token.")
        print(f"    Vui lòng chạy lại script kèm cờ --token <YOUR_HF_TOKEN> để tải các mô hình này.")

    # 5. Download Demucs
    print(f"\n[+] Đang tải mô hình: Demucs (htdemucs)...")
    try:
        import demucs.pretrained
        # Demucs tự động dùng thư mục TORCH_HOME đã set ở trên
        demucs.pretrained.get_model('htdemucs')
        print(f"    -> Xong!")
    except Exception as e:
        print(f"    -> Lỗi: {e}. Vui lòng chạy pip install demucs")

    # 6. Download PANNS (buộc phải lách bằng cách tạm tráo biến HOME)
    print(f"\n[+] Đang tải mô hình: PANNS (Audio tagging)...")
    try:
        # PANNS ngầm tải về thư mục ~/panns_data, ta lừa nó tải vào base_dir
        old_home = os.environ.get("HOME", "")
        os.environ["HOME"] = base_dir
        from panns_inference import AudioTagging
        # Hàm này sẽ trigger download
        AudioTagging(checkpoint_path=None, device='cpu')
        os.environ["HOME"] = old_home
        print(f"    -> Xong!")
    except Exception as e:
        print(f"    -> Lỗi tải PANNS: {e}. Vui lòng cài panns-inference")

    # 7. Download Sortformer (NeMo)
    print(f"\n[+] Đang tải mô hình: Sortformer (NeMo diar_msdd_telephonic)...")
    try:
        # NeMo dùng biến XDG_CACHE_HOME hoặc HOME
        os.environ["XDG_CACHE_HOME"] = base_dir
        from nemo.collections.asr.models import EncDecDiarLabelModel
        EncDecDiarLabelModel.from_pretrained("diar_msdd_telephonic")
        print(f"    -> Xong!")
    except Exception as e:
        print(f"    -> Lỗi tải NeMo Sortformer: {e}. Có thể bỏ qua nếu dùng Pyannote.")

    # 8. Download the BS-RoFormer vocal checkpoint (audio-separator).
    #    It does not come from the HF hub, so HF_HUB_OFFLINE does nothing for
    #    it: without this the music-removal stage is the one stage that still
    #    needs the network, and it fails deep into a run.
    sep_dir = os.path.join(base_dir, "audio-separator")
    os.makedirs(sep_dir, exist_ok=True)
    ckpt = os.environ.get("BS_ROFORMER_MODEL",
                          "model_bs_roformer_ep_368_sdr_12.9628.ckpt")
    print(f"\n[+] Đang tải mô hình: BS-RoFormer ({ckpt})...")
    try:
        from audio_separator.separator import Separator
        sep = Separator(model_file_dir=sep_dir, log_level=40)
        # download_model_files fetches the checkpoint and its config without
        # building the model; load_model is the fallback for releases that do
        # not expose it, and costs a model construction we throw away.
        # This also writes download_checks.json into sep_dir -- the remote
        # model list audio-separator fetches on every load. Without it the
        # stage still needs the network even with the weights already there.
        if hasattr(sep, "download_model_files"):
            sep.download_model_files(ckpt)
        else:
            sep.load_model(model_filename=ckpt)
        print(f"    -> Xong!")
    except Exception as e:
        print(f"    -> Lỗi tải BS-RoFormer: {e}. Cần `pip install audio-separator`.")

    print(f"\n=====================================================")
    print(f"TẢI HOÀN TẤT! Toàn bộ models đã nằm trong: {base_dir}")
    print(f"Để sử dụng chúng (Chạy Offline trên Kaggle), hãy set các biến môi trường sau:")
    print(f"export HF_HOME=\"{hf_cache}\"")
    print(f"export TORCH_HOME=\"{torch_cache}\"")
    print(f"export HOME=\"{base_dir}\"  # Dành cho PANNS")
    print(f"export XDG_CACHE_HOME=\"{base_dir}\"  # Dành cho NeMo")
    print(f"export BS_ROFORMER_MODEL_DIR=\"{sep_dir}\"  # Dành cho BS-RoFormer")
    print(f"=====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải toàn bộ Model Weights về máy tính")
    parser.add_argument("--token", type=str, help="HuggingFace Access Token", default=None)
    args = parser.parse_args()
    
    download_models(args.token)
