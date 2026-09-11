"""Song song hoá WindowPlanner.build() qua nhiều process, pipeline với GPU.

WindowPlanner.build() là CPU/numpy thuần -- không đụng GPU -- và với overlap
có nhiều support candidate có thể tốn vài giây một job (đo được 3.5s/job trên
kịch bản có lượt nói xen kẽ ngắn). Trong khi đó self.bss_model.separate_two_speakers()
chạy GPU tuần tự: không cách nào chạy nhiều job đó cùng lúc trên một GPU. Nhưng
việc BUILD (CPU) của job N+1 không phụ thuộc gì vào việc TÁCH (GPU) của job N,
nên có thể build song song trong lúc GPU đang bận -- đây là pipeline, không
phải chạy theo lô.

Thiết kế:
  - Một ProcessPoolExecutor sống suốt một lần gọi process_overlaps(), không
    tạo lại mỗi job -- chi phí spawn đo được ~0.5-0.8s/process trên máy dev.
  - waveform chia sẻ qua shared_memory thay vì pickle lại cho mỗi worker: một
    file dài có thể hàng trăm MB, nhân theo số worker là lãng phí RAM thật.
  - Mỗi worker tự dựng WindowPlanner MỘT LẦN (trong initializer), rồi build()
    nhiều job liên tiếp trên cùng planner đó.
  - use_vad mặc định TẮT. Từng thử cho mỗi worker tự tải Silero VAD CPU-only
    và gặp crash tầng native thật: nhiều process mới spawn cùng lúc gọi
    torch.hub.load() lần đầu (cache rỗng) đua nhau tải/giải nén, một số
    process chết với "libc++abi: recursive_mutex lock failed" -- không phải
    exception Python bắt được, ProcessPoolExecutor âm thầm thay worker chết
    và lượt chạy tiếp tục nhưng cho kết quả sai (đo được spliced=0 trên kịch
    bản 4 job). Rơi về energy-based cut trong worker (use_vad=False) tránh
    hẳn lớp rủi ro này; đây là hành vi sẵn có của AcousticCuts khi không có
    VAD, không phải lối thoát tạm. Cái giá thật: cửa sổ dựng song song cắt
    kém chính xác hơn một chút so với nhánh tuần tự (vốn dùng Silero GPU của
    chính bss_model). Bật lại use_vad=True chỉ nên làm sau khi pre-warm cache
    Silero ở main process TRƯỚC khi tạo pool (đảm bảo mọi worker gặp cache đã
    ấm, không đua tải) -- chưa được kiểm chứng đủ kỹ ở đây để làm mặc định.
  - Job được nộp hết một lượt (submit, không phải map) theo đúng thứ tự; đọc
    kết quả cũng theo thứ tự đó bằng future.result(). Trong lúc main thread
    đang block chờ future[i] để chạy GPU, các worker rảnh đã âm thầm build
    tiếp future[i+1..]. Đây là chỗ tạo hiệu ứng pipeline: CPU và GPU chồng lấp
    thời gian thay vì nối đuôi nhau. Đổi sang executor.map ở đây sẽ mất tính
    chất này vì map lô theo chunksize thay vì để mọi future chạy ngay khi có
    worker rảnh.

Phụ thuộc một bản vá trong separation_window.py: WindowPlanner.build() từng
nhận diện "cặp overlap của job này" bằng id() -- đúng khi mọi thứ ở chung một
process, sai ngay khi pairs/group đi qua pickle (id() đổi, build() tưởng nhầm
cặp của chính mình là cặp lạ, tự chặn mọi job). Module này không hoạt động
đúng nếu bản vá đó bị revert.
"""
import pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np

_worker = {}


def _load_cpu_vad():
    """Silero VAD ép chạy CPU, độc lập với device của bss_model.

    Không dùng lại instance GPU của bss_model._vad: onnxruntime.InferenceSession
    không picklable, và chia 5 process cùng một GPU context của một model đã
    load là nguồn lỗi/tranh chấp không đáng đánh đổi lấy vài trăm ms VAD mỗi
    job. Trả None nếu tải lỗi -- AcousticCuts tự rơi về energy-based cut,
    hành vi này đã có sẵn từ trước, không phải lối thoát mới.
    """
    try:
        import torch
        from models.silero_vad import SileroVAD
        return SileroVAD(device=torch.device("cpu"))
    except Exception:
        return None


def _init_worker(shm_name, shape, dtype_str, segs_bytes, pairs, music_map_bytes,
                  seams, sr, context_seconds, search_seconds, use_vad):
    """Chạy đúng một lần khi worker process khởi động. Dựng WindowPlanner và
    giữ trong biến module-level để _build_job dùng lại cho mọi job worker
    này nhận, không dựng lại mỗi lần."""
    from utils.separation_window import WindowPlanner

    shm = shared_memory.SharedMemory(name=shm_name)
    waveform = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    segments = pickle.loads(segs_bytes)
    music_map = pickle.loads(music_map_bytes) if music_map_bytes is not None else None
    vad = _load_cpu_vad() if use_vad else None

    _worker["planner"] = WindowPlanner(
        segments, pairs, waveform, sr, music_map=music_map, seams=seams,
        vad=vad, context_seconds=context_seconds, search_seconds=search_seconds)
    _worker["shm"] = shm  # giữ tham chiếu sống -- không để GC đóng shared memory


def _build_job(group):
    planner = _worker["planner"]
    result = planner.build(group)
    # planner.reason/detail chỉ có ý nghĩa khi result is None; kèm theo để
    # caller ghi log/lý do thất bại giống hệt nhánh tuần tự.
    return result, planner.reason, planner.detail


class WindowBuildPool:
    """Pool sống suốt một lần process_overlaps(); build() song song, pipeline với GPU.

    Dùng như context manager:
        with WindowBuildPool(segments, pairs, waveform, sr, ...) as pool:
            for built, reason, detail in pool.build_all(job_groups):
                ...  # chạy GPU ở đây trong lúc worker build job tiếp theo
    """

    def __init__(self, segments, pairs, waveform, sr, music_map=None, seams=(),
                 context_seconds=2.0, search_seconds=400.0, n_workers=4, use_vad=False):
        self.n_workers = max(1, int(n_workers))

        # shared_memory yêu cầu size > 0; một waveform rỗng (đường hiếm gặp,
        # test đơn vị) vẫn phải cấp phát được.
        self._shm = shared_memory.SharedMemory(create=True, size=max(1, waveform.nbytes))
        shm_arr = np.ndarray(waveform.shape, dtype=waveform.dtype, buffer=self._shm.buf)
        shm_arr[:] = waveform[:]

        segs_bytes = pickle.dumps(segments)
        music_map_bytes = pickle.dumps(music_map) if music_map is not None else None

        self._pool = ProcessPoolExecutor(
            max_workers=self.n_workers, initializer=_init_worker,
            initargs=(self._shm.name, waveform.shape, str(waveform.dtype), segs_bytes,
                      pairs, music_map_bytes, tuple(seams), sr, context_seconds,
                      search_seconds, use_vad),
        )
        self._closed = False

    def build_all(self, job_groups):
        """Nộp hết job_groups theo thứ tự, trả generator (built, reason, detail)
        cùng thứ tự. Nộp hết một lượt (không phải theo yêu cầu từng cái) để
        worker rảnh build tiếp job sau trong lúc caller còn xử lý (GPU) job
        trước -- chính chỗ này tạo hiệu ứng pipeline."""
        futures = [self._pool.submit(_build_job, group) for group in job_groups]
        for f in futures:
            yield f.result()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True)
        self._shm.close()
        self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
