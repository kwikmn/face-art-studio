"""Fail-closed live output, independent of Qt. No raw-frame fallback."""
from collections import Counter, deque
from dataclasses import dataclass
import threading
import sys
import time
import os

import cv2
import numpy as np


@dataclass(frozen=True)
class LiveConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    processing_fps: int = 20
    max_age: float = 0.25
    hold_last_good: float = 0.75


@dataclass(frozen=True)
class ProcessedFrame:
    image: np.ndarray | None
    reason: str = "Live"


def fit_frame(frame, width, height):
    """Letterbox, never crop a possible second face out of the input."""
    h, w = frame.shape[:2]
    if (w,h)==(width,height) and os.environ.get('FACEART_COPY_REFERENCE')!='1':
        return frame
    scale = min(width / w, height / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    result = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - size[0]) // 2, (height - size[1]) // 2
    result[y:y + size[1], x:x + size[0]] = resized
    return result


class LivePipeline:
    def __init__(self, capture, processor, output, config=None):
        self.config = config or LiveConfig()
        self.capture, self.processor, self.output = capture, processor, output
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._raw_ready = threading.Event()
        self._source_event = threading.Event()
        self._source_request = self._source_ready = None
        self._source_version = 0
        self._source_status = ""
        self._raw = self._safe = self._preview = None
        self._paused = self._capture_failed = False
        self._enabled = False
        self._generation = self._sequence = 0
        self._reason = "Starting"
        self._output_error = None
        self._threads = []
        self._counts = Counter()
        self._block_reasons = Counter()
        self._processing_ms, self._age_ms = deque(maxlen=900), deque(maxlen=900)
        self._safe_times = deque(maxlen=900)
        self._capture_times = deque(maxlen=900)
        self._started = time.perf_counter()
        c = self.config
        self.slate = np.full((c.height, c.width, 3), 24, dtype=np.uint8)
        cv2.putText(self.slate, "VIDEO PAUSED", (c.width // 8, c.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, c.width / 900),
                    (220, 220, 220), 2, cv2.LINE_AA)

    def start(self):
        if self._threads:
            raise RuntimeError("A live session can only be started once")
        self._started = time.perf_counter()
        # Output starts before slow camera/model initialization.
        for name, target in (("output", self._publish), ("capture", self._capture),
                             ("processing", self._process), ("source", self._load_source)):
            thread = threading.Thread(target=target, name=f"protected-{name}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def change_source(self, path):
        """Queue the latest selection without blocking capture or the UI."""
        with self._lock:
            if self._stop.is_set():
                return
            self._source_version += 1
            self._source_request = (self._source_version, str(path))
            self._source_ready = None
            self._source_status = "Loading new face; current face remains active"
            self._source_event.set()

    def _load_source(self):
        while not self._stop.is_set():
            self._source_event.wait(.1)
            with self._lock:
                request, self._source_request = self._source_request, None
                self._source_event.clear()
            if request is None or self._stop.is_set():
                continue
            version, path = request
            try:
                face = self.processor.prepare_source_image(path)
                error = None
            except Exception as exc:
                face, error = None, str(exc)
            with self._lock:
                if version != self._source_version or self._stop.is_set():
                    continue
                if error is not None:
                    self._source_status = f"Face change failed; previous face retained: {error}"
                else:
                    self._source_ready = (face, path)
                    self._source_status = "New face ready"

    def pause(self, paused=True):
        with self._lock:
            self._paused = paused
            self._generation += 1
            self._safe = self._raw = None
            self._reason = "Paused" if paused else "Waiting for a fresh face"

    def enable_output(self, enabled):
        with self._lock:
            self._enabled = enabled
            self._generation += 1
            self._safe = self._raw = None
            if enabled:
                self._output_error = None

    def _invalidate(self, reason, hard=False):
        with self._lock:
            # Brief failures repeat only the previous validated swap. Its original
            # timestamp is never extended, so repeated failures still expire.
            if hard:
                self._safe = None
                self._generation += 1
            self._reason = reason
            self._counts["blocked"] += 1
            self._block_reasons[reason] += 1

    def _capture(self):
        c = self.config
        try:
            if not self.capture.start(c.width, c.height, c.fps):
                raise RuntimeError("Cannot open the selected camera")
            next_frame = time.perf_counter()
            while not self._stop.is_set():
                ok, frame = self.capture.read()
                captured_at = time.perf_counter()
                if not ok or frame is None:
                    raise RuntimeError("Camera disconnected")
                if captured_at < next_frame:
                    continue
                next_frame = captured_at + 1 / c.fps * 0.95
                with self._lock:
                    self._counts["captured"] += 1
                    self._capture_times.append(captured_at)
                    if self._raw is not None:
                        self._counts["dropped_capture"] += 1
                    self._raw = (frame, captured_at, self._generation)
                    self._raw_ready.set()
        except Exception as error:
            with self._lock:
                self._capture_failed = True
            self._invalidate(str(error), hard=True)
        finally:
            self.capture.release()

    def _process(self):
        c = self.config
        try:
            self.processor.prepare()
            next_frame = time.perf_counter()
            while not self._stop.is_set():
                if self._stop.wait(max(0, next_frame - time.perf_counter())):
                    break
                self._raw_ready.wait(0.05)
                with self._lock:
                    if self._source_ready is not None:
                        face, path = self._source_ready
                        self.processor.activate_source(face, path)
                        self._source_ready = None
                        self._source_status = "Face changed"
                    packet, self._raw = self._raw, None
                    self._raw_ready.clear()
                    paused = self._paused
                if packet is None or paused:
                    continue
                frame, captured_at, generation = packet
                started = time.perf_counter()
                next_frame = started + 1 / c.processing_fps
                if started - captured_at > c.max_age:
                    self._invalidate("Camera frame too old")
                    continue
                try:
                    result = self.processor.process(frame)
                    if not isinstance(result, ProcessedFrame):
                        raise ValueError("Processor did not return a protected result")
                    image = result.image
                    if image is not None:
                        if (image.dtype != np.uint8 or image.ndim != 3 or
                                image.shape[2] != 3 or min(image.shape[:2]) == 0):
                            raise ValueError("Invalid processed frame")
                        image = fit_frame(image, c.width, c.height)
                except Exception as error:
                    self._invalidate(f"Processing failed: {error}")
                    continue
                finished = time.perf_counter()
                with self._lock:
                    self._processing_ms.append((finished - started) * 1000)
                    self._counts["processed"] += 1
                    if (generation != self._generation or self._paused or
                            self._capture_failed or self._stop.is_set()):
                        continue
                    if image is None or finished - captured_at > c.max_age:
                        self._reason = result.reason if image is None else "Processing too slow"
                        self._counts["blocked"] += 1
                        self._block_reasons[self._reason] += 1
                    else:
                        self._safe = (image, captured_at)
                        self._reason = "Live"
                        self._counts["safe_frames"] += 1
                        self._safe_times.append(finished)
        except Exception as error:
            self._invalidate(f"Model setup failed: {error}", hard=True)

    def _publish(self):
        c = self.config
        timer = None
        if sys.platform == 'win32':
            import ctypes
            timer = ctypes.WinDLL('winmm')
            if timer.timeBeginPeriod(1) != 0:
                timer = None
        try:
            while not self._stop.is_set():
                # Serializes pause/disable with the final send. No frame chosen
                # before a pause may be sent after pause() returns.
                with self._lock:
                    # Lock contention must not leave the freshness decision stale.
                    now = time.perf_counter()
                    safe = self._safe
                    fresh = (safe is not None and not self._paused and
                             not self._capture_failed and now - safe[1] <= c.hold_last_good)
                    frame = safe[0] if fresh else self.slate
                    if safe is not None and not fresh and not self._paused:
                        self._reason = "Waiting for a fresh processed frame"
                    self._preview = frame
                    self._sequence += 1
                    if fresh:
                        if now - safe[1] > c.max_age:
                            self._counts["held_frames"] += 1
                        self._age_ms.append((now - safe[1]) * 1000)
                    try:
                        if self._enabled:
                            self.output.send(frame)
                            self._counts["published"] += 1
                            if not fresh:
                                self._counts["slate_frames"] += 1
                        else:
                            self.output.close()
                    except Exception as error:
                        self._output_error = str(error)
                        self._enabled = False
                        self._safe = None
                        self.output.close()
                # Anchor each tick to its actual send/selection time. Delayed wakes
                # drop ticks instead of squeezing a catch-up send into the period.
                self._stop.wait(max(0, now + 1 / c.fps - time.perf_counter()))
        finally:
            try:
                if self._enabled:
                    self.output.send(self.slate)
            except Exception:
                pass
            finally:
                self.output.close()
                if timer is not None:
                    timer.timeEndPeriod(1)

    def preview(self):
        with self._lock:
            return self._sequence, self._preview

    def metrics(self):
        with self._lock:
            elapsed = max(0.001, time.perf_counter() - self._started)
            def percentile(values, pct):
                return round(float(np.percentile(list(values), pct)), 2) if values else None
            return {**self._counts, "block_reasons": dict(self._block_reasons), "seconds": round(elapsed, 2),
                    "processed_fps": round(self._counts["processed"] / elapsed, 2),
                    "recent_capture_fps": round(sum(t > time.perf_counter()-5 for t in self._capture_times) / min(5, elapsed), 2),
                    "safe_fps": round(self._counts["safe_frames"] / elapsed, 2),
                    "recent_safe_fps": round(sum(t > time.perf_counter()-5 for t in self._safe_times) / min(5, elapsed), 2),
                    "processing_p50_ms": percentile(self._processing_ms, 50),
                    "processing_p95_ms": percentile(self._processing_ms, 95),
                    "frame_age_p95_ms": percentile(self._age_ms, 95),
                    "status": self._reason, "output_error": self._output_error,
                    "output_device": self.output.info() if hasattr(self.output,'info') else None,
                    "output_enabled": self._enabled, "paused": self._paused,
                    "source_status": self._source_status}

    def stop(self, timeout=3):
        self.pause()
        self._stop.set()
        self._raw_ready.set()
        self._source_event.set()
        deadline = time.perf_counter() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - time.perf_counter()))
        return not any(thread.is_alive() for thread in self._threads)
