"""Progress of the refinement LLM pass, written to the log instead of a tqdm bar.

A tqdm bar rewrites one console line with `\\r`, so none of it reached the log
file. This writes ordinary log records, one writer, and the count only moves
forward: `[LLM] 120/450 (27%, 3.1 seg/s, ~110s)`.
"""

import threading
import time


class LlmProgress:
    def __init__(self, total, logger=None, label="[LLM]", unit="seg",
                 interval: float = 15.0, clock=time.monotonic, usage=None):
        self.total = max(0, int(total))
        self.logger = logger
        self.label = label
        self.unit = unit
        self.interval = float(interval)
        self._clock = clock
        self._usage = usage        # optional callable -> {"prompt_tokens", "completion_tokens"}
        self._lock = threading.Lock()
        self._done = 0
        self._started = clock()
        self._last_report = None
        self._last_done = -1

    @property
    def done(self) -> int:
        return self._done

    def advance(self, count: int = 1) -> None:
        with self._lock:
            self._done = min(self.total, self._done + max(0, int(count)))
        self.report()

    def render(self) -> str:
        with self._lock:
            done, now = self._done, self._clock()
        details = []
        if self.total:
            details.append(f"{100 * done // self.total}%")
        elapsed = now - self._started
        if done > 0 and elapsed > 0:
            rate = done / elapsed
            details.append(f"{rate:.1f} {self.unit}/s")
            if done < self.total:
                details.append(f"~{int((self.total - done) / rate)}s")
        text = f"{self.label} {done}/{self.total}" + (
            f" ({', '.join(details)})" if details else "")
        tokens = self._token_text()
        return text + (f" | {tokens}" if tokens else "")

    def _token_text(self) -> str:
        if self._usage is None:
            return ""
        try:
            usage = self._usage() or {}
        except Exception:
            return ""
        prompt = int(usage.get("prompt_tokens", 0))
        out = int(usage.get("completion_tokens", 0))
        if not prompt and not out:
            return ""
        return f"tokens in={prompt} out={out}"

    def report(self, force: bool = False) -> None:
        """Write a line when the interval has passed, or when the pass is complete."""
        now = self._clock()
        with self._lock:
            done = self._done
            complete = self.total and done >= self.total
            due = self._last_report is None or now - self._last_report >= self.interval
            if done == self._last_done and not force:
                return
            if not (force or complete or due):
                return
            self._last_report, self._last_done = now, done
        if self.logger:
            self.logger.info(self.render())

    def finish(self) -> None:
        if self._last_done != self._done:
            self.report(force=True)
