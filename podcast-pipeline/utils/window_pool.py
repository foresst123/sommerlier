"""Song song hoá WindowPlanner.build() qua nhiều process, pipeline với GPU,
pool sống suốt cả batch nhiều file, không phải tạo lại cho từng file.

WindowPlanner.build() là CPU/numpy thuần -- không đụng GPU -- và với overlap
có nhiều support candidate có thể tốn vài giây một job. Trong khi đó
self.bss_model.separate_two_speakers() chạy GPU tuần tự: không cách nào chạy
nhiều job đó cùng lúc trên một GPU. Nhưng việc BUILD (CPU) của job N+1 không
phụ thuộc gì vào việc TÁCH (GPU) của job N, nên có thể build song song trong
lúc GPU đang bận -- đây là pipeline, không phải chạy theo lô.

Bài học từ lần đo trên Kaggle thật: pool từng được tạo/đóng MỖI FILE (bên
trong process_overlaps). Chi phí spawn process (đo được ~22.87s cho window
đầu tiên, 8 worker/4 core) không chia sẻ được giữa các file -- một batch N
file trả chi phí đó N lần. WindowBuildPool ở đây tách làm hai tầng để sửa
đúng chỗ đó:

  - WindowBuildPool: pool process, KHÔNG chứa dữ liệu riêng của file nào.
    Tạo một lần, dùng cho mọi file trong cả batch, đóng một lần ở cuối stage
    (PipelineService gọi qua begin_stage_scope()/end_stage_scope(), cùng chỗ
    Sidon worker được giữ sống qua nhiều file).
  - FileWindows (trả về từ pool.open_file()): một file cụ thể. Tạo
    shared_memory RIÊNG cho waveform của file đó (không tránh được -- mỗi
    file dữ liệu khác nhau), đóng lại khi file đó xử lý xong. Không đụng tới
    pool process bên dưới.

Vì mỗi job giờ tự dựng WindowPlanner ngay trong lúc chạy (không còn qua
initializer -- initializer chỉ chạy một lần lúc worker khởi động, không có
cách gọi lại nó cho file mới mà không giết rồi tạo lại worker), có một đánh
đổi nhỏ: WindowPlanner.__init__() (dựng by_speaker, clean_segments) chạy lại
mỗi job thay vì một lần mỗi (worker, file). Với vài trăm segment việc này mất
cỡ mili giây -- không đáng kể cạnh chi phí spawn process đã tránh được, và
đổi lại tránh hẳn việc phải "báo" cho đủ mọi worker biết file đã đổi (một bài
toán đồng bộ không tầm thường: ProcessPoolExecutor không cho nhắm task vào
đúng worker cụ thể).

Silero VAD: use_vad mặc định TẮT. Từng thử cho mỗi worker tự tải Silero VAD
CPU-only và gặp crash tầng native thật: nhiều process mới spawn cùng lúc gọi
torch.hub.load() lần đầu (cache rỗng) đua nhau tải/giải nén, một số process
chết với "libc++abi: recursive_mutex lock failed" -- không phải exception
Python bắt được, ProcessPoolExecutor âm thầm thay worker chết và lượt chạy
tiếp tục nhưng cho kết quả sai (đo được spliced=0 trên kịch bản 4 job). Rơi
về energy-based cut trong worker (use_vad=False) tránh hẳn lớp rủi ro này --
hành vi sẵn có của AcousticCuts khi không có VAD, không phải lối thoát tạm.
Cái giá thật: cửa sổ dựng song song cắt kém chính xác hơn một chút so với
nhánh tuần tự (vốn dùng Silero GPU của chính bss_model).

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


def _load_cpu_vad():
    """Silero VAD ép chạy CPU, độc lập với device của bss_model.

    Không dùng lại instance GPU của bss_model._vad: onnxruntime.InferenceSession
    không picklable, và chia nhiều process cùng một GPU context của một model
    đã load là nguồn lỗi/tranh chấp không đáng đánh đổi lấy vài trăm ms VAD
    mỗi job. Trả None nếu tải lỗi -- AcousticCuts tự rơi về energy-based cut.
    """
    try:
        import torch
        from models.silero_vad import SileroVAD
        return SileroVAD(device=torch.device("cpu"))
    except Exception:
        return None


def _build_job(file_ctx, group):
    """Chạy trong worker process. Tự đứng hoàn toàn: gắn shared memory theo
    tên, dựng WindowPlanner riêng cho lần gọi này, build() job, rồi thả
    shared memory trước khi trả kết quả -- không giữ gì lại giữa các job,
    vì worker này còn phục vụ job của những file khác trong suốt vòng đời
    của pool."""
    from utils.separation_window import WindowPlanner

    shm = shared_memory.SharedMemory(name=file_ctx["shm_name"])
    try:
        waveform = np.ndarray(file_ctx["shape"], dtype=np.dtype(file_ctx["dtype"]),
                              buffer=shm.buf)
        segments = pickle.loads(file_ctx["segs_bytes"])
        music_map_bytes = file_ctx["music_map_bytes"]
        music_map = pickle.loads(music_map_bytes) if music_map_bytes is not None else None
        vad = _load_cpu_vad() if file_ctx["use_vad"] else None

        planner = WindowPlanner(
            segments, file_ctx["pairs"], waveform, file_ctx["sr"],
            music_map=music_map, seams=file_ctx["seams"], vad=vad,
            context_seconds=file_ctx["context_seconds"],
            search_seconds=file_ctx["search_seconds"])
        result = planner.build(group)
        # planner.reason/detail chỉ có ý nghĩa khi result is None; kèm theo để
        # caller ghi log/lý do thất bại giống hệt nhánh tuần tự.
        return result, planner.reason, planner.detail
    finally:
        # Chỉ đóng mapping trong worker này (unmap cục bộ); KHÔNG unlink --
        # đó là việc của FileWindows.close() ở main process, sau khi mọi job
        # của file này đã có kết quả.
        shm.close()


class FileWindows:
    """Một file cụ thể trên một WindowBuildPool đã có sẵn.

    Tạo qua WindowBuildPool.open_file(); build_all() nộp job của CHÍNH FILE
    NÀY vào pool dùng chung, close() chỉ giải phóng shared memory của file
    này -- không đụng tới pool process (pool sống tiếp cho file kế)."""

    def __init__(self, pool_executor, segments, pairs, waveform, sr,
                 music_map=None, seams=(), context_seconds=2.0,
                 search_seconds=400.0, use_vad=False):
        self._pool = pool_executor

        # shared_memory yêu cầu size > 0; một waveform rỗng (đường hiếm gặp,
        # test đơn vị) vẫn phải cấp phát được.
        self._shm = shared_memory.SharedMemory(create=True, size=max(1, waveform.nbytes))
        shm_arr = np.ndarray(waveform.shape, dtype=waveform.dtype, buffer=self._shm.buf)
        shm_arr[:] = waveform[:]

        self._ctx = {
            "shm_name": self._shm.name,
            "shape": waveform.shape,
            "dtype": str(waveform.dtype),
            "segs_bytes": pickle.dumps(segments),
            "pairs": pairs,
            "music_map_bytes": pickle.dumps(music_map) if music_map is not None else None,
            "seams": tuple(seams),
            "sr": sr,
            "context_seconds": context_seconds,
            "search_seconds": search_seconds,
            "use_vad": use_vad,
        }
        self._closed = False

    def build_all(self, job_groups):
        """Nộp hết job_groups của file này theo thứ tự, trả generator
        (built, reason, detail) cùng thứ tự. Nộp hết một lượt (không phải
        theo yêu cầu từng cái) để worker rảnh build tiếp job sau trong lúc
        caller còn xử lý (GPU) job trước -- chính chỗ này tạo hiệu ứng
        pipeline. Đổi sang executor.map ở đây sẽ mất tính chất này vì map lô
        theo chunksize thay vì để mọi future chạy ngay khi có worker rảnh."""
        futures = [self._pool.submit(_build_job, self._ctx, group) for group in job_groups]
        for f in futures:
            yield f.result()

    def close(self):
        """Giải phóng shared memory của RIÊNG file này. Không shutdown pool --
        pool còn phục vụ file tiếp theo trong batch."""
        if self._closed:
            return
        self._closed = True
        self._shm.close()
        self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class WindowBuildPool:
    """Pool process dùng chung cho CẢ BATCH nhiều file -- tạo một lần, đóng
    một lần ở cuối stage, không phải mỗi file. Không giữ dữ liệu riêng của
    file nào; mỗi file gọi open_file() để lấy một FileWindows.

    Dùng như:
        pool = WindowBuildPool(n_workers=4)
        for file in batch:
            with pool.open_file(segments, pairs, waveform, sr, ...) as fw:
                for built, reason, detail in fw.build_all(job_groups):
                    ...  # chạy GPU ở đây trong lúc worker build job tiếp theo
        pool.close()   # một lần, sau khi xử lý xong cả batch
    """

    def __init__(self, n_workers=4):
        self.n_workers = max(1, int(n_workers))
        # Không cần initializer: mỗi job tự đứng hoàn toàn (_build_job nhận
        # đủ ngữ cảnh qua file_ctx), nên không có state nào phải dựng trước
        # lúc worker khởi động.
        self._pool = ProcessPoolExecutor(max_workers=self.n_workers)
        self._closed = False

    def open_file(self, segments, pairs, waveform, sr, music_map=None, seams=(),
                  context_seconds=2.0, search_seconds=400.0, use_vad=False):
        return FileWindows(
            self._pool, segments, pairs, waveform, sr, music_map=music_map,
            seams=seams, context_seconds=context_seconds,
            search_seconds=search_seconds, use_vad=use_vad)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
