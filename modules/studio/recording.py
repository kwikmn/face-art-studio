"""Protected-frame recording. The recorder never receives physical-camera frames."""

from pathlib import Path
import json
import locale
import re
import shutil
import subprocess
import threading
import time

import numpy as np

HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def executable(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} is not installed or is missing from PATH")
    return found


def audio_devices():
    result = subprocess.run(
        [
            executable("ffmpeg"),
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
        ],
        capture_output=True,
        creationflags=HIDDEN,
        timeout=15,
    )
    try:
        listing = result.stderr.decode("utf-8")
    except UnicodeDecodeError:
        listing = result.stderr.decode(
            locale.getpreferredencoding(False), errors="replace"
        )
    return re.findall(r'"([^"\r\n]+)" \(audio\)', listing)


def probe(path):
    result = subprocess.run(
        [
            executable("ffprobe"),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        creationflags=HIDDEN,
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-1200:])
    return json.loads(result.stdout)


class Recorder:
    """One paced writer, bounded memory, NVENC video and selected audio in one muxer."""

    def __init__(
        self,
        frame_provider,
        width,
        height,
        path,
        microphone="",
        test_audio=False,
        audio_delay_ms=0,
    ):
        if (
            isinstance(audio_delay_ms, bool)
            or not isinstance(audio_delay_ms, (int, np.integer))
            or not 0 <= audio_delay_ms <= 2000
        ):
            raise ValueError("Audio delay must be a whole number from 0 to 2000 ms")
        self.audio_delay_ms = int(audio_delay_ms)
        self.provider = frame_provider
        self.width, self.height = width, height
        self.path = Path(path)
        self.microphone, self.test_audio = microphone, test_audio
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None
        self.error = None
        self.frames = 0
        self.started = None
        self.finished = False

    def start(self):
        if self.path.exists():
            raise FileExistsError(f"Recording already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(
            target=self._record, name="studio-recorder", daemon=True
        )
        self.thread.start()

    def _command(self):
        cmd = [executable("ffmpeg"), "-hide_banner", "-loglevel", "warning", "-n"]
        audio = bool(self.microphone or self.test_audio)
        if self.test_audio:
            cmd += ["-re", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
        elif self.microphone:
            cmd += [
                "-thread_queue_size",
                "512",
                "-f",
                "dshow",
                "-audio_buffer_size",
                "50",
                "-i",
                f"audio={self.microphone}",
            ]
        cmd += [
            "-thread_queue_size",
            "8",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-map",
            "1:v:0" if audio else "0:v:0",
        ]
        if audio:
            filters = "aresample=async=1000:first_pts=0"
            if self.audio_delay_ms:
                # Normalize capture timestamps first, then delay all channels.
                filters += f",adelay=delays={self.audio_delay_ms}:all=1"
            cmd += [
                "-map",
                "0:a:0",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-af",
                filters,
                "-metadata",
                f"FACEART_AUDIO_DELAY_MS={self.audio_delay_ms}",
                "-shortest",
            ]
        cmd += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "ll",
            "-rc",
            "vbr",
            "-cq",
            "23",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
            str(self.path),
        ]
        return cmd

    def _record(self):
        log_path = self.path.with_suffix(".recording.log")
        try:
            with log_path.open("wb") as log:
                self.process = subprocess.Popen(
                    self._command(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    creationflags=HIDDEN,
                )
                self.started = time.perf_counter()
                while not self.stop_event.is_set():
                    deadline = self.started + self.frames / 30
                    if self.stop_event.wait(max(0, deadline - time.perf_counter())):
                        break
                    if time.perf_counter() - deadline > 2:
                        raise RuntimeError(
                            "Recording could not keep up; the partial clip was retained"
                        )
                    _, frame = self.provider()
                    if frame is None:
                        frame = np.full((self.height, self.width, 3), 24, np.uint8)
                    if (
                        frame.shape != (self.height, self.width, 3)
                        or frame.dtype != np.uint8
                    ):
                        raise ValueError("Recorder rejected invalid protected output")
                    self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
                    self.frames += 1
                self.process.stdin.close()
                if self.process.wait(timeout=15):
                    raise RuntimeError("FFmpeg could not finish the recording")
            if not self.frames:
                raise RuntimeError("No frames were recorded")
        except Exception as exc:
            self.error = str(exc)
            if self.process and self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            if log_path.exists():
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                if detail.strip():
                    self.error += "\n" + detail
        finally:
            self.finished = True

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(18)
            if self.thread.is_alive():
                self.error = "Recorder stopped responding; partial clip retained"
                if self.process and self.process.poll() is None:
                    self.process.kill()
                self.thread.join(3)
        return self.path
