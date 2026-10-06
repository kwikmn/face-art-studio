import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from modules.studio.postwork import export_media
from modules.studio.recording import Recorder, audio_devices


class StudioMediaTests(unittest.TestCase):
    def test_audio_device_names_preserve_windows_accented_characters(self):
        from types import SimpleNamespace

        listing = '[dshow] "Mikrofon (gränssnitt)" (audio)'.encode("cp1252")
        with (
            patch("modules.studio.recording.executable", return_value="ffmpeg"),
            patch(
                "modules.studio.recording.subprocess.run",
                return_value=SimpleNamespace(stderr=listing),
            ),
            patch(
                "modules.studio.recording.locale.getpreferredencoding",
                return_value="cp1252",
            ),
        ):
            self.assertEqual(audio_devices(), ["Mikrofon (gränssnitt)"])

    def test_recording_never_overwrites_an_existing_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "saved.mkv"
            path.write_bytes(b"original recording")
            recorder = Recorder(lambda: (0, None), 80, 60, path)
            with self.assertRaises(FileExistsError):
                recorder.start()
            self.assertEqual(path.read_bytes(), b"original recording")

    def test_recording_rejects_invalid_frame_without_writing_it(self):
        class Process:
            def __init__(self):
                self.stdin = io.BytesIO()
                self.returncode = None

            def poll(self):
                return self.returncode

            def kill(self):
                self.returncode = 1

            def wait(self, *a, **k):
                self.returncode = 1
                return 1

        process = Process()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("modules.studio.recording.executable", return_value="ffmpeg"),
            patch("modules.studio.recording.subprocess.Popen", return_value=process),
        ):
            recorder = Recorder(
                lambda: (1, np.zeros((60, 80, 3), dtype=np.float32)),
                80,
                60,
                Path(folder) / "test.mkv",
            )
            recorder.start()
            recorder.thread.join(2)
            self.assertIn("invalid protected output", recorder.error)
            self.assertEqual(process.stdin.getvalue(), b"")
            self.assertTrue(recorder.finished)

    def test_postwork_rejects_source_as_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "original.png"
            path.write_bytes(b"original")
            with self.assertRaises(ValueError):
                export_media(path, path, None, 0.6, lambda *_: None)
            self.assertEqual(path.read_bytes(), b"original")

    def test_cancelled_image_export_does_not_publish_a_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.png"
            output = Path(folder) / "export.png"
            ok, data = cv2.imencode(".png", np.full((32, 32, 3), 123, np.uint8))
            self.assertTrue(ok)
            data.tofile(str(source))
            original = source.read_bytes()
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(InterruptedError):
                export_media(source, output, None, 0.6, lambda *_: None, cancel)
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
