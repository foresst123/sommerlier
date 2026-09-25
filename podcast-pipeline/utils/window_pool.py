"""Song song hoá WindowPlanner.build_many() qua nhiều process, pipeline với GPU,
pool sống suốt cả batch nhiều file, không phải tạo lại cho từng file.

WindowPlanner.build_many() là CPU/numpy thuần -- không đụng GPU -- và với overlap
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
hành vi sẵn có của AcousticBoundaryFinder khi không có VAD.
Cái giá thật: cửa sổ dựng song song cắt kém chính xác hơn một chút so với
nhánh tuần tự (vốn dùng Silero GPU của chính bss_model).

Phụ thuộc một bản vá trong separation_window.py: WindowPlanner.build() từng
nhận diện "cặp overlap của job này" bằng id() -- đúng khi mọi thứ ở chung một
process, sai ngay khi pairs/group đi qua pickle (id() đổi, build() tưởng nhầm
cặp của chính mình là cặp lạ, tự chặn mọi job). Module này không hoạt động
đúng nếu bản vá đó bị revert.
"""
import os
import pickle
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import shared_memory

import numpy as np


# Rough bytes a window holds while in flight: its audio, the two raw Sidon
# tracks, and the two assigned tracks of the result waiting for the consumer.
ADMISSION_COST_FACTOR = 5.0
# Share of the soft RAM limit the separation lookahead may use when the budget
# is derived (admission_bytes = 0).
ADMISSION_RAM_SHARE = 0.25


def window_cost_bytes(built) -> int:
    """Estimated in-flight bytes of one built window (0 when it failed to build)."""
    audio = getattr(built, "audio", None)
    nbytes = getattr(audio, "nbytes", None)
    if nbytes is None:
        return 0
    return int(nbytes * ADMISSION_COST_FACTOR)


def total_ram_bytes() -> int:
    """Physical RAM, or 0 when it cannot be read."""
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 0


def resolve_admission_bytes(configured, ram_soft_fraction=0.75) -> int:
    """Byte budget for windows in flight. <0 = off (0 returned), 0 = derive from
    ram_soft_fraction x physical RAM x ADMISSION_RAM_SHARE, >0 = as given."""
    configured = int(configured)
    if configured < 0:
        return 0
    if configured > 0:
        return configured
    return int(total_ram_bytes() * float(ram_soft_fraction) * ADMISSION_RAM_SHARE)


class ByteBudget:
    """Admission ledger, not a blocker: fits() says whether one more item may be
    taken now, charge()/release() move the bytes. limit <= 0 means unlimited.
    An item is always allowed when nothing is in flight, so one oversized
    window cannot stall the stream."""

    def __init__(self, limit=0):
        self.limit = max(0, int(limit))
        self.in_flight = 0
        self._lock = threading.Lock()

    def fits(self, cost) -> bool:
        with self._lock:
            return (self.limit == 0 or self.in_flight == 0
                    or self.in_flight + cost <= self.limit)

    def charge(self, cost):
        with self._lock:
            self.in_flight += int(cost)

    def release(self, cost):
        with self._lock:
            self.in_flight = max(0, self.in_flight - int(cost))


def _load_cpu_vad():
    """Silero VAD ép chạy CPU, độc lập với device của bss_model.

    Không dùng lại instance GPU của bss_model._vad: onnxruntime.InferenceSession
    không picklable, và chia nhiều process cùng một GPU context của một model
    đã load là nguồn lỗi/tranh chấp không đáng đánh đổi lấy vài trăm ms VAD
    mỗi job. Trả None nếu tải lỗi -- boundary finder tự rơi về energy-based cut.
    """
    try:
        import torch
        from models.silero_vad import SileroVAD
        return SileroVAD(device=torch.device("cpu"))
    except Exception:
        return None


def _build_job(file_ctx, group):
    """Chạy trong worker process. Tự đứng hoàn toàn: gắn shared memory theo
    tên, dựng WindowPlanner riêng cho lần gọi này, build_many() job, rồi thả
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

        planner = None
        try:
            planner = WindowPlanner(
                segments, file_ctx["pairs"], waveform, file_ctx["sr"],
                music_map=music_map, seams=file_ctx["seams"], vad=vad,
                context_seconds=file_ctx["context_seconds"],
                max_context_seconds=file_ctx["max_context_seconds"],
                padding_min_seconds=file_ctx["padding_min_seconds"],
                search_seconds=file_ctx["search_seconds"])
            result = planner.build_many(group)
            return result, planner.reason, planner.detail, list(planner.actions)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            actions = list(getattr(planner, "actions", []))
            actions.append({
                "step": len(actions) + 1,
                "action": "window_builder_exception",
                "detail": detail,
                "traceback": traceback.format_exc(),
            })
            return None, "window_error", detail, actions
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

    # Mức nộp trước mặc định: nộp lần lượt. Đặt ở cấp lớp để một thể hiện dựng
    # bằng object.__new__ (test dựng trần để khỏi cấp shared memory) vẫn chạy
    # build_all được; __init__ ghi đè bằng giá trị của pool.
    _max_pending = 1

    def __init__(self, pool_executor, segments, pairs, waveform, sr,
                 music_map=None, seams=(), context_seconds=2.0,
                 max_context_seconds=2.2, padding_min_seconds=1.0,
                 search_seconds=400.0, use_vad=False, max_pending=None):
        self._pool = pool_executor
        self._max_pending = max(1, int(max_pending or 1))

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
            "max_context_seconds": max_context_seconds,
            "padding_min_seconds": padding_min_seconds,
            "search_seconds": search_seconds,
            "use_vad": use_vad,
        }
        self._closed = False
        self._futures = []

    def build_all(self, job_groups):
        """Nộp hết job_groups của file này theo thứ tự, trả generator
        (built, reason, detail, actions) cùng thứ tự. Nộp hết một lượt (không phải
        theo yêu cầu từng cái) để worker rảnh build tiếp job sau trong lúc
        caller còn xử lý (GPU) job trước -- chính chỗ này tạo hiệu ứng
        pipeline. Đổi sang executor.map ở đây sẽ mất tính chất này vì map lô
        theo chunksize thay vì để mọi future chạy ngay khi có worker rảnh."""
        groups = iter(job_groups)
        pending = []

        def submit_next():
            try:
                group = next(groups)
            except StopIteration:
                return False
            try:
                future = self._pool.submit(_build_job, self._ctx, group)
                self._futures.append(future)
                pending.append((group, future, None))
            except Exception as exc:
                pending.append((group, None, f"{type(exc).__name__}: {exc}"))
            return True

        for _ in range(self._max_pending):
            if not submit_next():
                break
        while pending:
            group, future, error = pending.pop(0)
            if future is not None:
                try:
                    result = future.result()
                    submit_next()
                    yield result
                    continue
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            submit_next()
            action = {"action": "worker_fallback_sequential", "detail": error}
            try:
                built, reason, detail, actions = _build_job(self._ctx, group)
                actions = [action, *actions]
                for plan in built or []:
                    plan.actions.insert(0, dict(action))
                    if plan.window is not None:
                        plan.window.layout["actions"].insert(0, dict(action))
                yield built, reason, detail, actions
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                yield None, "window_error", detail, [action, {
                    "action": "sequential_fallback_exception", "detail": detail,
                    "traceback": traceback.format_exc(),
                }]

    def close(self):
        """Giải phóng shared memory của RIÊNG file này. Không shutdown pool --
        pool còn phục vụ file tiếp theo trong batch."""
        if self._closed:
            return
        self._closed = True
        for future in self._futures:
            future.cancel()
        if self._futures:
            wait(self._futures)
        self._futures.clear()
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
                for built, reason, detail, actions in fw.build_all(job_groups):
                    ...  # chạy GPU ở đây trong lúc worker build job tiếp theo
        pool.close()   # một lần, sau khi xử lý xong cả batch
    """

    def __init__(self, n_workers=4, max_pending=None):
        self.n_workers = max(1, int(n_workers))
        self.max_pending = max(1, int(max_pending or self.n_workers * 2))
        # Không cần initializer: mỗi job tự đứng hoàn toàn (_build_job nhận
        # đủ ngữ cảnh qua file_ctx), nên không có state nào phải dựng trước
        # lúc worker khởi động.
        self._pool = ProcessPoolExecutor(max_workers=self.n_workers)
        self._closed = False

    def open_file(self, segments, pairs, waveform, sr, music_map=None, seams=(),
                  context_seconds=2.0, max_context_seconds=2.2,
                  padding_min_seconds=1.0, search_seconds=400.0, use_vad=False):
        return FileWindows(
            self._pool, segments, pairs, waveform, sr, music_map=music_map,
            seams=seams, context_seconds=context_seconds,
            max_context_seconds=max_context_seconds,
            padding_min_seconds=padding_min_seconds,
            search_seconds=search_seconds, use_vad=use_vad,
            max_pending=self.max_pending)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
