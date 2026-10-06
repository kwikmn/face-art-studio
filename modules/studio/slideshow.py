"""Bounded background-only playback. Image I/O never runs in video/UI ticks."""

from collections import OrderedDict, deque
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_image(path, size):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Image exceeds 32 MB limit")
    with Image.open(path) as image:
        if image.width * image.height > 24_000_000 or max(image.size) > 20000:
            raise ValueError("Image exceeds 24 megapixel / 20000 edge limit")
        image = ImageOps.exif_transpose(image)
        image = ImageOps.fit(
            image.convert("RGB"), size, method=Image.Resampling.LANCZOS
        )
        result = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
        result.flags.writeable = False
        return result


class ImageStore:
    """Three resized cache entries, three queued keys, one decoding worker."""

    def __init__(self, decoder=decode_image):
        self.decoder = decoder
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.queue = deque()
        self.active = None
        self.event = threading.Event()
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, name="background-slides-loader", daemon=True
        )
        self.thread.start()

    def get(self, key):
        with self.lock:
            value = self.cache.get(key)
            if key in self.cache:
                self.cache.move_to_end(key)
            return value

    def request(self, key, priority=False):
        with self.lock:
            if self.closed or key in self.cache or key == self.active:
                return
            if key in self.queue:
                self.queue.remove(key)
            if priority:
                self.queue.appendleft(key)
            else:
                self.queue.append(key)
            while len(self.queue) > 3:
                self.queue.pop()
            self.event.set()

    def _run(self):
        while True:
            self.event.wait()
            with self.lock:
                if self.closed:
                    return
                if not self.queue:
                    self.event.clear()
                    continue
                key = self.active = self.queue.popleft()
            try:
                value = (self.decoder(key[0], key[1]), None)
            except Exception as exc:
                value = (None, f"{Path(key[0]).name}: {exc}")
            with self.lock:
                self.active = None
                if self.closed:
                    return
                self.cache[key] = value
                self.cache.move_to_end(key)
                while len(self.cache) > 3:
                    self.cache.popitem(last=False)

    def close(self):
        with self.lock:
            self.closed = True
            self.queue.clear()
            self.cache.clear()
            self.event.set()
        # Current decode is bounded but must not stall camera shutdown/UI.


class Slideshow:
    def __init__(self, settings=None, store=None, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.store = store or ImageStore()
        saved = settings if isinstance(settings, dict) else {}
        paths = saved.get("paths", [])
        paths = paths[:200] if isinstance(paths, list) else []
        self.paths = list(dict.fromkeys(x for x in paths if isinstance(x, str)))
        self.active = (
            saved.get("active")
            if saved.get("active") in self.paths
            else (self.paths[0] if self.paths else None)
        )
        self.pending = self.active
        self.current = self.old = None
        self.size = (1280, 720)

        def number(key, default, low, high):
            try:
                value = float(saved.get(key, default))
            except (ValueError, TypeError):
                value = default
            if not np.isfinite(value):
                value = default
            return max(low, min(high, value))

        self.interval = number("interval", 10, 1, 3600)
        self.fade = number("fade", 0.35, 0, 2)
        self.loop = bool(saved.get("loop", True))
        self.playing = False
        self.started = self.clock()
        self.deadline = self.started + self.interval
        self.paused_at = self.started
        self.remaining = self.interval
        self.error = ""
        self.closed = False
        self.frozen = False
        self.cursor = self.active

    def settings(self):
        with self.lock:
            # Persist a surviving requested selection, while keeping the last
            # validated pixels in memory until that selection finishes loading.
            selected = (
                self.pending
                if self.pending in self.paths
                else self.active
                if self.active in self.paths
                else self.cursor
                if self.cursor in self.paths
                else self.paths[0]
                if self.paths
                else None
            )
            return dict(
                paths=self.paths.copy(),
                active=selected,
                interval=self.interval,
                fade=self.fade,
                loop=self.loop,
            )

    def set_size(self, width, height):
        with self.lock:
            size = (int(width), int(height))
            if size != self.size:
                self.size = size
                self.current = self.old = None
                self.pending = self.active

    def set_paths(self, paths):
        with self.lock:
            previous = self.paths
            index = previous.index(self.active) if self.active in previous else 0
            self.paths = list(dict.fromkeys(str(x) for x in paths))[:200]
            if not self.paths:
                self.active = self.pending = None
                self.current = self.old = None
                self.playing = False
                self.error = "Playlist empty"
                return
            if self.active not in self.paths:
                self.pending = self.paths[min(index, len(self.paths) - 1)]
                self.cursor = self.pending
            if self.pending not in self.paths:
                self.pending = (
                    self.active if self.active in self.paths else self.paths[0]
                )

    def select(self, path):
        with self.lock:
            if path in self.paths:
                self.pending = path
                self.cursor = path
                self.error = ""

    def navigate(self, step):
        with self.lock:
            if not self.paths:
                return
            base = (
                self.pending
                if self.pending in self.paths
                else (self.cursor if self.cursor in self.paths else self.active)
            )
            index = self.paths.index(base) if base in self.paths else 0
            target = (
                (index + step) % len(self.paths)
                if self.loop
                else max(0, min(len(self.paths) - 1, index + step))
            )
            self.select(self.paths[target])

    def play(self, enabled):
        with self.lock:
            now = self.clock()
            if enabled == self.playing:
                return
            if enabled:
                if not self.paths:
                    return
                if self.frozen:
                    self.started += now - self.paused_at
                self.frozen = False
                self.deadline = now + self.remaining
            else:
                self.remaining = max(0, self.deadline - now)
                self.paused_at = now
                self.frozen = self.old is not None
            self.playing = enabled

    def configure(self, interval, fade, loop):
        with self.lock:
            interval = max(1, min(3600, float(interval)))
            self.fade = max(0, min(2, float(fade)))
            self.loop = bool(loop)
            if interval != self.interval:
                self.interval = interval
                self.remaining = interval
                self.deadline = self.clock() + interval

    def tick(self):
        with self.lock:
            if self.closed:
                return
            now = self.clock()
            if self.playing and self.pending is None and now >= self.deadline:
                if (
                    self.cursor in self.paths
                    and self.paths.index(self.cursor) == len(self.paths) - 1
                    and not self.loop
                ):
                    self.play(False)
                else:
                    self.navigate(1)
                # Drop missed intervals: never burst through an unloaded playlist.
                self.deadline = now + self.interval
            if self.pending:
                key = (self.pending, self.size)
                value = self.store.get(key)
                if value is None:
                    self.store.request(key, True)
                else:
                    image, error = value
                    if error:
                        self.error = error
                        self.pending = None
                        self.deadline = now + self.interval
                    else:
                        # Rapid navigation cuts from latest validated background;
                        # never retain an unbounded chain of blended frames.
                        if self.pending != self.active or self.current is None:
                            self.old = self.current
                            self.current = image
                            self.active = self.pending
                            self.cursor = self.active
                            self.started = now
                            self.paused_at = now
                            self.frozen = False
                        self.pending = None
                        self.error = ""
                        self.deadline = now + self.interval
                        self.remaining = self.interval
            if self.active in self.paths:
                index = self.paths.index(self.active)
                if index + 1 < len(self.paths) or self.loop:
                    self.store.request(
                        (self.paths[(index + 1) % len(self.paths)], self.size)
                    )
            if (
                self.old is not None
                and (self.paused_at if self.frozen else now) - self.started >= self.fade
            ):
                self.old = None

    def frame(self, width, height):
        # No decoding, resizing or waiting for the loader here.
        with self.lock:
            if (width, height) != self.size:
                return None
            current, old = self.current, self.old
            elapsed = (self.paused_at if self.frozen else self.clock()) - self.started
            fraction = 1 if self.fade == 0 else max(0, min(1, elapsed / self.fade))
        if current is None:
            return None
        return (
            current
            if old is None or fraction >= 1
            else cv2.addWeighted(old, 1 - fraction, current, fraction, 0)
        )

    def status(self):
        with self.lock:
            name = Path(self.active).name if self.active else "No active image"
            return (
                ("Playing" if self.playing else "Paused")
                + " · "
                + name
                + (" · Loading " + Path(self.pending).name if self.pending else "")
                + (" · " + self.error if self.error else "")
            )

    def close(self):
        with self.lock:
            self.closed = True
            self.current = self.old = None
        self.store.close()
