"""Đo độ lệch thời gian giữa track đã tách và mixture đưa vào Sidon.

Chạy trên các file _dump_tracks đã ghi sẵn, nên không cần chạy lại model:

    python scripts/measure_sidon_lag.py output/<audio>/03_separation/audio/raw/separated

Mỗi cửa sổ có <tag>_mix.wav, <tag>_trackA.wav, <tag>_trackB.wav. So tổng hai
track với mixture: cái model trả về là cùng nội dung, chia làm hai, nên tổng
của chúng phải trùng thời điểm với mixture.

So bằng ENVELOPE năng lượng chứ không bằng mẫu thô. Sidon dựng lại sóng âm qua
VAE nên pha không giữ nguyên -- chính sidon_infer._channel_similarity đã đo
được lệch 2ms là đủ kéo tương quan mẫu thô từ 1.0 xuống -0.64. Envelope không
đổi dưới phép dịch đó, và cái ta cần biết ở đây là "lời nói rơi vào lúc nào",
không phải "sóng âm nằm ở đâu".

Kỳ vọng sau khi sidon_infer bỏ phần đệm đầu: lệch ~0ms. Nếu vẫn còn một giá trị
khác 0 lặp lại đều trên mọi cửa sổ, đó là quy ước khung của model, trừ thêm
đúng bấy nhiêu vào PAD_SAMPLES_IN.
"""
import os
import sys

import numpy as np

FRAME = 240          # 10ms tại 24kHz; đủ ngắn để bám âm tiết
SEARCH_MS = 120.0    # Chỉ tìm trong khoảng này; xa hơn là lỗi khác, không phải lệch pha


def envelope(x, frame=FRAME):
    n = len(x) // frame
    if n < 2:
        return None
    return np.sqrt((x[:n * frame].reshape(n, frame) ** 2).mean(axis=1))


def lag_ms(mix, sep, sr):
    """Số ms mà `sep` đi sau `mix`. Dương = trả về muộn."""
    a, b = envelope(sep), envelope(mix)
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a, b = a - a.mean(), b - b.mean()
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return None
    corr = np.correlate(a, b, "full") / (np.linalg.norm(a) * np.linalg.norm(b))
    centre = len(b) - 1
    reach = max(1, int(SEARCH_MS / 1000.0 * sr / FRAME))
    lo, hi = max(0, centre - reach), min(len(corr), centre + reach + 1)
    best = lo + int(np.argmax(corr[lo:hi]))
    return (best - centre) * FRAME / sr * 1000.0, float(corr[best])


def main(folder):
    import soundfile as sf

    tags = sorted(f[:-8] for f in os.listdir(folder) if f.endswith("_mix.wav"))
    if not tags:
        print(f"Không có file *_mix.wav nào trong {folder}")
        return 1

    rows = []
    for tag in tags:
        mix, sr = sf.read(os.path.join(folder, f"{tag}_mix.wav"), dtype="float32")
        try:
            a, _ = sf.read(os.path.join(folder, f"{tag}_trackA.wav"), dtype="float32")
            b, _ = sf.read(os.path.join(folder, f"{tag}_trackB.wav"), dtype="float32")
        except Exception:
            continue
        n = min(len(mix), len(a), len(b))
        got = lag_ms(mix[:n], a[:n] + b[:n], sr)
        if got is None:
            continue
        ms, score = got
        rows.append(ms)
        print(f"  {tag:<28} {ms:+7.1f} ms   (r={score:.3f})")

    if not rows:
        print("Không đo được cửa sổ nào (quá ngắn hoặc im lặng).")
        return 1

    arr = np.array(rows)
    print(f"\n{len(arr)} cửa sổ | trung vị {np.median(arr):+.1f} ms | "
          f"trung bình {arr.mean():+.1f} ms | lệch chuẩn {arr.std():.1f} ms")
    print(f"khoảng [{arr.min():+.1f}, {arr.max():+.1f}] ms")
    if abs(np.median(arr)) < FRAME / 24000 * 1000:
        print("=> Trong một khung. Không còn lệch hệ thống.")
    else:
        print(f"=> Còn lệch đều {np.median(arr):+.1f} ms. Chỉnh PAD_SAMPLES_IN "
              f"trong sidon_infer.py thêm {np.median(arr) / 1000 * 16000:+.0f} mẫu.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
