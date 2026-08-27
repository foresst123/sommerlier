"""Chạy Sidon trên CẢ FILE thay vì từng cửa sổ ghép quanh mỗi overlap.

Pipeline hiện gọi Sidon trên cửa sổ ghép ~11 giây dựng quanh từng overlap được
phát hiện, tức khoảng 1% audio. Tách cả file sẽ cho thứ tự kênh nhất quán suốt
bản ghi: không phải gán lại track bằng ECAPA cho từng job, và không phụ thuộc
vào việc diarization có nhìn thấy overlap hay không. Log cho thấy sự phụ thuộc
đó đang làm mất backchannel thật:

    overlap 0.64% is below 3.0%: the diarizer is probably missing backchannels

Ba câu hỏi còn bỏ ngỏ: VRAM có đủ không, cơ chế _maybe_swap của Sidon có giữ
được thứ tự kênh qua ~200 chunk thay vì 1 chunk không, và backchannel nhỏ hơn
giọng chính 15 dB có sống sót khi bị chuẩn hoá theo chunk 20 giây thay vì theo
cửa sổ vốn được dựng cho cân bằng không.

Không câu nào trả lời được bằng cách đọc code. File này chạy thí nghiệm và ghi
hai track ra đĩa để nghe.

    python tools/try_whole_file_sidon.py audio.mp3 --minutes 3
    python tools/try_whole_file_sidon.py audio.mp3            # cả file

Bắt đầu bằng --minutes: 3 phút không vừa thì 50 phút chắc chắn không, và chạy
ngắn cho biết điều đó sau một phút thay vì hai mươi.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_audio(path, sr, minutes=None):
    from services.audio_service import AudioService
    audio = AudioService().load_audio(path, target_sr=sr)
    wav = np.asarray(audio.waveform, dtype=np.float32)
    if minutes:
        wav = wav[: int(minutes * 60 * sr)]
    return wav


def run_worker(python_bin, wav, sr, workdir, logger=print):
    """Điều khiển sidon_worker qua giao thức stdin/stdout, đúng một yêu cầu."""
    npy = os.path.join(workdir, "mix.npy")
    np.save(npy, wav)

    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    proc = subprocess.Popen(
        [python_bin, "sidon_worker.py"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env,
    )

    # Chờ dòng "ready" rồi mới gửi việc.
    while True:
        line = proc.stdout.readline()
        if not line:
            err = proc.stderr.read()
            raise RuntimeError(f"worker chết trước khi sẵn sàng:\n{err[-2000:]}")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        logger(f"  worker: {msg}")
        if msg.get("status") == "ready":
            break

    t0 = time.time()
    proc.stdin.write(json.dumps({"id": "1", "audio_path": npy, "sample_rate": sr}) + "\n")
    proc.stdin.flush()

    resp = None
    while True:
        line = proc.stdout.readline()
        if not line:
            err = proc.stderr.read()
            raise RuntimeError(f"worker chết giữa lúc tách:\n{err[-2000:]}")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == "1":
            resp = msg
            break
    elapsed = time.time() - t0

    proc.stdin.close()
    proc.terminate()
    return resp, elapsed, proc.stderr.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--minutes", type=float, default=None,
                    help="chỉ tách N phút đầu (nên bắt đầu nhỏ)")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--out", default="whole_file_sidon")
    args = ap.parse_args()

    from utils.worker_env import resolve_worker_python
    cfg = json.load(open("config.json"))
    python_bin = resolve_worker_python("sidon", config=cfg, env_profile={})

    print(f"audio      {args.audio}")
    wav = load_audio(args.audio, args.sr, args.minutes)
    dur = len(wav) / args.sr
    print(f"độ dài     {dur:.0f}s ({dur/60:.1f} min) @ {args.sr}Hz")
    print(f"worker     {python_bin}\n")

    os.makedirs(args.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            resp, elapsed, stderr = run_worker(python_bin, wav, args.sr, tmp)
        except RuntimeError as e:
            print(f"\nHỎNG: {e}")
            print("\nNếu đây là CUDA out of memory thì chính nó là câu trả lời: VRAM ở")
            print("đây tăng theo độ dài, nên hạ --minutes để tìm ngưỡng.")
            return 1

        if resp.get("error"):
            print(f"\nlỗi từ worker: {resp['error']}")
            print(stderr[-2000:])
            return 1

        # Worker trả về ĐƯỜNG DẪN chứ không phải mảng, và đặt tên khoá là
        # track_1_path / track_2_path.
        t1 = np.load(resp["track_1_path"])
        t2 = np.load(resp["track_2_path"])
        out_sr = int(resp.get("target_sr", args.sr))

    import soundfile as sf
    base = os.path.splitext(os.path.basename(args.audio))[0]
    tag = f"{base}_{int(dur)}s"
    paths = []
    for name, arr in (("mix", wav), ("trackA", t1), ("trackB", t2)):
        p = os.path.join(args.out, f"{tag}_{name}.wav")
        sf.write(p, arr, args.sr if name == "mix" else out_sr)
        paths.append(p)

    def rms(a):
        return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))

    print(f"\nTách xong trong {elapsed:.1f}s  ({elapsed/dur:.2f}x thời gian thực)")
    print(f"  mix     rms {rms(wav):.4f}")
    print(f"  track A rms {rms(t1):.4f}")
    print(f"  track B rms {rms(t2):.4f}")

    # Một track gần như im lặng chính là lỗi "trả về một nguồn và im lặng" mà
    # pipeline đã đếm dưới tên empty_track, nhưng ở quy mô cả file. Biết trước
    # thì đỡ mất công ngồi nghe.
    quiet = [n for n, a in (("A", t1), ("B", t2)) if rms(a) < rms(wav) * 0.02]
    if quiet:
        print(f"\n  !! track {'/'.join(quiet)} gần như im lặng — bộ tách đã trả về")
        print("     một nguồn và không có gì cho nguồn còn lại.")

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
        print(f"\n  GPU lúc này: {smi}")
    except FileNotFoundError:
        pass

    print("\nĐã ghi:")
    for p in paths:
        print(f"  {p}")
    print("\nNghe hai track từ đầu đến cuối. Câu hỏi không phải là chúng có sạch")
    print("không, mà là track A có giữ nguyên MỘT người suốt cả file không.")
    print("Nếu giọng đổi giữa chừng thì _maybe_swap không trụ được ở độ dài này,")
    print("và cửa sổ ghép theo từng overlap vẫn là lựa chọn đúng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())