"""Kiểm tra chất lượng và tính nhất quán danh tính speaker
cho các đoạn audio đã được tách ra từ overlap window.

Mục tiêu:
- Kiểm tra track tách ra có đủ năng lượng âm thanh không.
- Kiểm tra track hiện tại có bị đổi speaker so với window trước không.
"""

import numpy as np


def track_quality(host, track, sr, floor=0.002, frame_sec=0.02):
    """
    Kiểm tra chất lượng của một track đã tách.

    host:
        Audio gốc / mixture trong đoạn cần kiểm tra.

    track:
        Audio đã tách ra cho một speaker.

    sr:
        Sample rate, ví dụ 16000 hoặc 24000.

    floor:
        Ngưỡng RMS tối thiểu.
        Nếu năng lượng nhỏ hơn mức này thì coi như gần như im lặng.

    frame_sec:
        Độ dài mỗi frame để chia nhỏ audio ra kiểm tra.
        Mặc định 0.02 giây = 20ms.
    """

    # Ép audio về numpy array dạng float64 và làm phẳng thành 1 chiều
    host = np.asarray(host, dtype=np.float64).reshape(-1)
    track = np.asarray(track, dtype=np.float64).reshape(-1)

    # Chỉ so sánh trên phần độ dài chung của host và track
    n = min(len(host), len(track))

    # Kết quả mặc định
    result = {
        "samples": n,
        "floor_rms": float(floor),
        "coverage": None,
        # Số sample trong mỗi frame, ví dụ 0.02 * 16000 = 320 sample
        "frame_samples": max(1, round(frame_sec * sr)),
    }

    # Nếu track rỗng hoặc track ngắn hơn host thì không đạt
    if n == 0 or len(track) < len(host):
        return dict(result, accepted=False, status="short_track")

    # Nếu audio có NaN hoặc inf thì không hợp lệ
    if not np.isfinite(host).all() or not np.isfinite(track).all():
        return dict(result, accepted=False, status="nonfinite_audio")

    frame = result["frame_samples"]

    # Các vị trí bắt đầu frame: 0, frame, 2*frame, ...
    starts = np.arange(0, n, frame)

    # Độ dài thật của từng frame, frame cuối có thể ngắn hơn
    counts = np.minimum(frame, n - starts)

    # RMS từng frame của audio gốc
    h = np.sqrt(np.add.reduceat(host[:n] ** 2, starts) / counts)

    # RMS từng frame của track đã tách
    t = np.sqrt(np.add.reduceat(track[:n] ** 2, starts) / counts)

    # Frame nào của host có tiếng thì coi là voiced
    # Điều kiện:
    # - RMS >= 25% RMS lớn nhất của host
    # - hoặc tối thiểu phải >= floor
    voiced = h >= max(float(h.max()) * 0.25, floor)

    # Lưu thêm thông tin để debug
    result.update(
        host_rms=float(np.sqrt(np.mean(host[:n] ** 2))),
        track_rms=float(np.sqrt(np.mean(track[:n] ** 2))),
        host_frame_rms=h.tolist(),
        track_frame_rms=t.tolist(),
        frame_lengths=counts.tolist(),
        voiced_frames=voiced.tolist(),
    )

    # Nếu host gần như không có tiếng thì chấp nhận
    # Vì mixture im lặng thì track im lặng cũng không sai
    if not voiced.any():
        return dict(result, accepted=True, status="quiet_mixture")

    # Frame nào của track còn sống / có năng lượng
    # Dùng 5% RMS lớn nhất của track hoặc floor
    alive = t >= max(float(t.max()) * 0.05, floor)

    # Tính tỷ lệ phần voiced của host mà track cũng có tiếng
    coverage = float(np.sum(counts[voiced & alive]) / np.sum(counts[voiced]))
    result["coverage"] = coverage

    # Nếu track có năng lượng trên ít nhất 10% vùng host có tiếng thì chấp nhận
    if coverage >= 0.10:
        return dict(result, accepted=True, status="energy_present")

    # Nếu đoạn quá ngắn thì chưa đủ bằng chứng để kết luận
    status = "insufficient_evidence" if n < round(0.1 * sr) else "low_energy_coverage"

    return dict(result, accepted=False, status=status)


def base_view(window, tracks):
    """
    Lấy phần base thật trong window và cắt track tương ứng.

    Window có thể gồm nhiều piece:
    - context trái
    - base
    - context phải

    Hàm này chỉ lấy phần kind == "base".
    """

    # Tìm piece chính có kind là "base"
    base = next(piece for piece in window.layout["pieces"] if piece["kind"] == "base")

    # Lấy vị trí của base trong danh sách pieces
    index = window.layout["pieces"].index(base)

    # Lấy số sample crossfade giữa các piece
    fades = window.layout.get("join_crossfade_samples")

    # Nếu không có join_crossfade_samples thì dùng crossfade_samples chung
    if fades is None:
        fades = [window.layout.get("crossfade_samples", 0)] * (
            len(window.layout["pieces"]) - 1
        )

    # Crossfade bên trái base
    left = fades[index - 1] if index else 0

    # Crossfade bên phải base
    right = fades[index] if index < len(fades) else 0

    # Vị trí sample gốc của base trong audio nguồn
    lo, hi = base["source_samples"]

    # Vị trí base nằm trong window
    offset = base["window_samples"][0]

    # Trả về:
    # 1. bounds trong audio gốc sau khi bỏ crossfade
    # 2. các track đã cắt đúng vùng base
    return (lo + left, hi - right), tuple(
        np.asarray(track[offset + left : offset + hi - lo - right]).copy()
        for track in tracks
    )


def continuity_evidence(previous, window, tracks, sr, floor=0.002):
    """
    Kiểm tra xem track hiện tại có bị đảo speaker so với window trước không.

    Ý tưởng:
    - Lấy vùng base hiện tại.
    - Tìm phần sample giao nhau với window trước.
    - Tính correlation giữa track cũ và track mới.
    - Nếu track cũ 0 giống track mới 1 hơn, và track cũ 1 giống track mới 0 hơn,
      thì có thể speaker đã bị swap.
    """

    # Lấy bounds và track của phần base hiện tại
    bounds, current = base_view(window, tracks)

    # Kết quả mặc định: chưa có đủ context để so sánh
    result = {
        "status": "no_shared_context",
        "swap": False,
        "margin": None,
    }

    # Nếu không có window trước thì không so được
    if previous is None:
        return result

    # Tìm đoạn giao nhau giữa base hiện tại và base trước
    lo = max(bounds[0], previous["bounds"][0])
    hi = min(bounds[1], previous["bounds"][1])

    # Nếu phần giao nhau ngắn hơn 0.1 giây thì bỏ qua
    if hi - lo < round(0.1 * sr):
        return result

    # Cắt track cũ đúng vùng giao nhau
    old = [
        t[lo - previous["bounds"][0] : hi - previous["bounds"][0]]
        for t in previous["tracks"]
    ]

    # Cắt track mới đúng vùng giao nhau
    new = [
        t[lo - bounds[0] : hi - bounds[0]]
        for t in current
    ]

    def correlation(a, b):
        """
        Tính độ tương quan giữa 2 đoạn audio.

        Giá trị càng cao thì 2 đoạn càng giống nhau.
        Dùng abs vì chỉ cần giống waveform, không quan tâm dấu âm/dương.
        """

        # Nếu độ dài khác nhau hoặc rỗng thì không tính
        if len(a) != len(b) or not len(a):
            return None

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # Nếu có NaN hoặc inf thì bỏ
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return None

        # Nếu một trong hai đoạn quá nhỏ năng lượng thì bỏ
        if min(np.sqrt(np.mean(a * a)), np.sqrt(np.mean(b * b))) < floor:
            return None

        # Trừ mean để correlation không bị lệch DC offset
        a = a - a.mean()
        b = b - b.mean()

        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))

        # Nếu mẫu đủ tốt thì tính correlation
        return abs(float(np.dot(a, b) / denominator)) if denominator > 1e-12 else None

    # Ma trận correlation:
    # matrix[old_track][new_track]
    matrix = [[correlation(a, b) for b in new] for a in old]

    # Trường hợp không swap:
    # old 0 khớp new 0
    # old 1 khớp new 1
    direct = [matrix[0][0], matrix[1][1]]

    # Trường hợp bị swap:
    # old 0 khớp new 1
    # old 1 khớp new 0
    swapped = [matrix[0][1], matrix[1][0]]

    # Tổng điểm khớp nếu không swap
    direct_score = sum(v or 0.0 for v in direct)

    # Tổng điểm khớp nếu swap
    swap_score = sum(v or 0.0 for v in swapped)

    # Nếu margin > 0 nghĩa là hướng swap có vẻ đúng hơn
    margin = swap_score - direct_score

    # Điều kiện đủ tin cậy:
    # - một trong hai hướng có tổng correlation >= 0.3
    # - chênh lệch giữa hai hướng >= 0.1
    enough = max(direct_score, swap_score) >= 0.3 and abs(margin) >= 0.1

    return dict(
        result,
        status="measured" if enough else "ambiguous",
        source_samples=[lo, hi],
        correlations=matrix,
        margin=margin,
        swap=bool(enough and margin > 0),
    )