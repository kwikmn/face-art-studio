"""Explicit raw webcam reference capture, separate from protected live output."""

import base64
import json
import sys
import time
import uuid
from pathlib import Path
from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel
from modules.studio.process_utils import stop_worker


class ReferenceCamera(QDialog):
    def __init__(self, index, name, folder, parent=None):
        super().__init__(parent)
        from modules.studio.app import label, button

        self.folder = Path(folder)
        self.camera_name = name
        self.path = None
        self.frame = None
        self.snapshot = None
        self.frame_time = 0
        self.buffer = b""
        self.process = QProcess(self)
        executable = Path(sys.executable)
        if executable.name.lower() == "pythonw.exe":
            executable = executable.with_name("python.exe")
        self.process.setProgram(str(executable))
        self.process.setArguments(
            ["-u", "-m", "modules.studio.reference_camera_worker", str(index)]
        )
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        self.process.readyReadStandardOutput.connect(self.read_frames)
        self.process.readyReadStandardError.connect(
            lambda: self.process.readAllStandardError()
        )
        self.process.errorOccurred.connect(self.camera_error)
        self.process.finished.connect(self.camera_stopped)
        self.setWindowTitle("Webcam reference · original camera image")
        self.resize(680, 780)
        self.setStyleSheet("QDialog {background:#101619;}")
        layout = QVBoxLayout(self)
        layout.addWidget(label("Webcam reference", "title"))
        layout.addWidget(
            label(
                "Original camera image · private reference preview. Nothing is sent to the virtual webcam.",
                "status",
            )
        )
        layout.addWidget(
            label(
                "Keep your entire head and beard inside the square. The photo is saved locally only when you choose Use snapshot.",
                "muted",
            )
        )
        self.preview = QLabel("Opening " + name + "…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 320)
        layout.addWidget(self.preview, 1)
        self.status = label("Opening camera…", "muted")
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.snap_button = button("Take photo", self.snap, "primary")
        self.retake_button = button("Retake", self.start)
        self.use_button = button("Use snapshot", self.use, "primary")
        row.addWidget(self.snap_button)
        row.addWidget(self.retake_button)
        row.addWidget(self.use_button)
        layout.addLayout(row)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_freshness)
        self.timer.start(250)
        self.start()

    def camera_error(self, *_):
        if self.snapshot is None:
            self.status.setText("Camera worker: " + self.process.errorString())
            self.retake_button.setEnabled(True)

    def camera_stopped(self, *_):
        if self.snapshot is None:
            self.frame = None
            self.snap_button.setEnabled(False)
            self.retake_button.setEnabled(True)
            if self.status.text().startswith(("Opening", "Original camera preview")):
                self.status.setText(
                    "Camera stopped. Check the selected device, then Retake to reconnect."
                )

    def start(self):
        stop_worker(self.process)
        self.snapshot = None
        self.frame = None
        self.buffer = b""
        self.snap_button.setEnabled(False)
        self.use_button.setEnabled(False)
        self.retake_button.setEnabled(False)
        self.status.setText("Opening " + self.camera_name + "…")
        self.process.start()

    def read_frames(self):
        self.buffer += bytes(self.process.readAllStandardOutput())
        latest = None
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            if not line.startswith(b"REFERENCE:"):
                continue
            try:
                data = json.loads(line[10:])
                if "error" in data:
                    self.status.setText(data["error"])
                    self.retake_button.setEnabled(True)
                elif self.snapshot is None:
                    latest = data.get("frame")
            except (ValueError, TypeError):
                self.status.setText("Cannot read a reference-camera frame.")
        if latest and self.snapshot is None:
            image = QImage.fromData(base64.b64decode(latest), "PNG")
            if image.isNull():
                return
            self.frame = image
            self.frame_time = time.monotonic()
            self.preview.setPixmap(
                QPixmap.fromImage(image).scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.status.setText("Original camera preview · ready to take a photo")
            self.snap_button.setEnabled(True)

    def check_freshness(self):
        if (
            self.snapshot is None
            and self.frame is not None
            and time.monotonic() - self.frame_time > 1
        ):
            self.snap_button.setEnabled(False)
            self.retake_button.setEnabled(True)
            self.status.setText("No fresh camera frame. Retake to reconnect.")

    def snap(self):
        if self.frame is None or time.monotonic() - self.frame_time > 1:
            return
        self.snapshot = self.frame.copy()
        stop_worker(self.process)
        self.snap_button.setEnabled(False)
        self.retake_button.setEnabled(True)
        self.use_button.setEnabled(True)
        self.status.setText("Photo taken · camera released. Use it or retake.")

    def use(self):
        if self.snapshot is None:
            return
        path = self.folder / f"reference-{uuid.uuid4().hex}.png"
        temporary = path.with_suffix(".tmp")
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            self.snapshot.setText(
                "FaceArt reference", "Original webcam snapshot · " + self.camera_name
            )
            if not self.snapshot.save(str(temporary), "PNG"):
                raise OSError("Check available disk space and folder permissions.")
            temporary.replace(path)
        except OSError as error:
            self.status.setText("Could not save the snapshot: " + str(error))
            return
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        self.path = str(path)
        self.accept()

    def done(self, result):
        self.timer.stop()
        stop_worker(self.process)
        super().done(result)
