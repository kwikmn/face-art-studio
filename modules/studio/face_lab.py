"""Face Lab dialog with an isolated, reusable local generation worker."""

import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QPlainTextEdit,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QFileDialog,
    QLabel,
    QAbstractItemView,
)
from modules.studio.facegen import validate_request
from modules.studio.process_utils import stop_worker
from modules.studio.snapshot_blend import file_hash


class FaceLab(QDialog):
    def __init__(self, studio):
        super().__init__(studio)
        from modules.studio.app import label, button, SETTINGS

        self.studio = studio
        self.output = SETTINGS.parent / "faces"
        self.reference = None
        self.selected = None
        self.busy = False
        self.buffer = b""
        self.errors = ""
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        executable = Path(sys.executable)
        if executable.name.lower() == "pythonw.exe":
            executable = executable.with_name("python.exe")
        self.process.setProgram(str(executable))
        self.process.setArguments(["-u", "-m", "modules.studio.facegen_worker"])
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.readyReadStandardError.connect(self.read_errors)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.setWindowTitle("FaceArt Studio · Face Lab")
        area = studio.screen().availableGeometry()
        self.resize(min(1100, area.width() - 40), min(820, area.height() - 60))
        self.setStyleSheet(
            "QDialog {background:#101619;} QPlainTextEdit, QLineEdit, QSpinBox, QListWidget {background:#192226;color:#e7eeed;border:1px solid #344149;border-radius:8px;padding:8px;}"
        )
        root = QVBoxLayout(self)
        title = QHBoxLayout()
        title.addWidget(label("Face Lab", "title"))
        title.addStretch()
        title.addWidget(label("FLUX.2 KLEIN 4B · LOCAL GENERATION", "eyebrow"))
        root.addLayout(title)
        root.addWidget(
            label(
                "Imagine a new face. Refine it with a photo. Choose when to use it.",
                "muted",
            )
        )
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(["Text to image", "Image to image"])
        self.mode.currentIndexChanged.connect(self.mode_changed)
        left.addWidget(self.mode)
        self.ref_preview = QLabel("No reference photo")
        self.ref_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ref_preview.setFixedSize(280, 130)
        left.addWidget(self.ref_preview)
        ref_row = QHBoxLayout()
        self.import_ref = button("Choose photo", self.choose_reference)
        self.current_ref = button(
            "Use current face", lambda: self.set_reference(studio.source)
        )
        ref_row.addWidget(self.import_ref)
        ref_row.addWidget(self.current_ref)
        left.addLayout(ref_row)
        self.webcam_ref = button("Take webcam snapshot", self.capture_reference)
        left.addWidget(self.webcam_ref)
        left.addWidget(label("Describe your face or the changes you want", "heading"))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(
            "A realistic headshot of a fictional adult with…\n\nFor edits: Give this person bushy eyebrows and a big red nose."
        )
        self.prompt.setPlainText(
            studio.settings.get(
                "facegen_prompt",
                "Photorealistic headshot of a fictional adult with warm brown eyes, wavy dark hair and a friendly expression. One person facing the camera, entire head visible, soft studio lighting, neutral background.",
            )
        )
        left.addWidget(self.prompt, 1)
        options = QHBoxLayout()
        self.size = QComboBox()
        for text, value in [
            ("512 · quick", 512),
            ("768 · balanced", 768),
            ("1024 · detailed", 1024),
        ]:
            self.size.addItem(text, value)
        self.size.setCurrentIndex(1)
        options.addWidget(self.size)
        self.count = QSpinBox()
        self.count.setRange(1, 4)
        self.count.setSuffix(" candidates")
        options.addWidget(self.count)
        left.addLayout(options)
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("Seed · leave blank for a new face")
        left.addWidget(self.seed)
        self.memory = QComboBox()
        for text, value in [
            ("Automatic memory use", "auto"),
            ("Full GPU · most VRAM", "gpu"),
            ("RAM offload · whole models", "model"),
            ("RAM offload · minimal VRAM, slower", "sequential"),
        ]:
            self.memory.addItem(text, value)
        index = self.memory.findData(studio.settings.get("facegen_memory", "auto"))
        self.memory.setCurrentIndex(max(0, index))
        left.addWidget(self.memory)
        left.addWidget(
            label(
                "Four-step generation. Models stay loaded while Face Lab is open; closing it releases GPU memory.",
                "muted",
            )
        )
        actions = QHBoxLayout()
        self.generate_button = button("Generate faces", self.generate, "primary")
        self.cancel_button = button("Cancel", self.cancel)
        self.cancel_button.setEnabled(False)
        actions.addWidget(self.generate_button)
        actions.addWidget(self.cancel_button)
        left.addLayout(actions)
        body.addLayout(left, 2)
        right = QVBoxLayout()
        self.preview = QLabel("Your next identity starts here")
        self.preview.setMinimumSize(320, 320)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#151d21;border-radius:12px;")
        right.addWidget(self.preview, 1)
        self.details = label("Select a candidate to preview it.", "muted")
        right.addWidget(self.details)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.setIconSize(QSize(90, 90))
        self.gallery.setMaximumHeight(175)
        self.gallery.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.gallery.currentItemChanged.connect(self.select)
        right.addWidget(self.gallery)
        self.use_button = button("Use this face", self.use_face, "primary")
        self.use_button.setEnabled(False)
        refine_row = QHBoxLayout()
        self.refine_button = button("Refine selected", self.refine_selected)
        self.settings_button = button("Reuse prompt / seed", self.reuse_settings)
        refine_row.addWidget(self.refine_button)
        refine_row.addWidget(self.settings_button)
        right.addLayout(refine_row)
        self.blend_button = button("Blend with original reference", self.blend_selected)
        right.addWidget(self.blend_button)
        right.addWidget(self.use_button)
        body.addLayout(right, 3)
        root.addLayout(body, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)
        self.status = label(
            "Ready · Images and prompts stay on this computer.", "status"
        )
        root.addWidget(self.status)
        self.mode_changed()
        self.set_busy(False)
        if self.output.exists():
            for path in sorted(
                self.output.glob("face-*.png"), key=lambda p: p.stat().st_mtime
            )[-40:]:
                self.add_result(str(path))

    def mode_changed(self):
        editing = self.mode.currentIndex() == 1
        for widget in (self.ref_preview, self.import_ref, self.current_ref):
            widget.setVisible(editing)

    def choose_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose reference photo",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            self.set_reference(path)

    def set_reference(self, path):
        pixmap = QPixmap(str(path)) if path else QPixmap()
        if pixmap.isNull():
            self.status.setText("Cannot open that reference photo.")
            return
        self.reference = str(path)
        try:
            digest = file_hash(path)
            references = dict(self.studio.settings.get("facegen_references", {}))
            references[digest] = self.reference
            self.studio.settings["facegen_references"] = dict(
                list(references.items())[-60:]
            )
            self.studio._save()
        except OSError as error:
            self.status.setText("Cannot remember reference photo: " + str(error))
        self.ref_preview.setPixmap(
            pixmap.scaled(
                280,
                130,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.mode.setCurrentIndex(1)

    def set_busy(self, busy):
        self.busy = busy
        self.generate_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.use_button.setEnabled(not busy and bool(self.selected))
        self.refine_button.setEnabled(not busy and bool(self.selected))
        self.settings_button.setEnabled(not busy and bool(self.selected))
        self.webcam_ref.setEnabled(not busy)
        self.blend_button.setEnabled(
            not busy
            and bool(getattr(self, "selected_metadata", {}).get("reference_sha256"))
        )
        for widget in (
            self.prompt,
            self.seed,
            self.size,
            self.count,
            self.memory,
            self.mode,
            self.import_ref,
            self.current_ref,
        ):
            widget.setEnabled(not busy)

    def generate(self):
        try:
            if self.mode.currentIndex() and not self.reference:
                raise ValueError(
                    "Choose a reference photo for image-to-image generation."
                )
            request = validate_request(
                dict(
                    prompt=self.prompt.toPlainText(),
                    size=self.size.currentData(),
                    count=self.count.value(),
                    memory=self.memory.currentData(),
                    seed=self.seed.text().strip() or None,
                    reference=self.reference if self.mode.currentIndex() else None,
                )
            )
        except ValueError as error:
            self.status.setText(str(error))
            return
        request["output_dir"] = str(self.output)
        self.studio.settings.update(
            facegen_prompt=request["prompt"], facegen_memory=request["memory"]
        )
        self.studio._save()
        self.set_busy(True)
        self.progress.setRange(0, 0)
        self.status.setText(
            "Starting Face Lab… First use downloads the model if needed."
        )
        self.errors = ""
        self.buffer = b""
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.process.start()
            if not self.process.waitForStarted(3000):
                self.set_busy(False)
                self.status.setText(
                    "Could not start the generation worker: "
                    + self.process.errorString()
                )
                return
        self.process.write((json.dumps(request) + "\n").encode("utf-8"))

    def read_errors(self):
        self.errors = (
            self.errors
            + bytes(self.process.readAllStandardError()).decode(
                "utf-8", errors="replace"
            )
        )[-6000:]

    def read_output(self):
        self.buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            if not line.startswith(b"FACELAB:"):
                continue
            try:
                event = json.loads(line[8:])
            except ValueError:
                continue
            kind = event.get("kind")
            if kind == "result":
                self.add_result(event["path"])
            elif kind == "progress":
                self.progress.setRange(0, 100)
                self.progress.setValue(event["value"])
            elif kind in ("done", "error"):
                self.set_busy(False)
                self.progress.setRange(0, 100)
                self.progress.setValue(100 if kind == "done" else 0)
            if "message" in event:
                self.status.setText(event["message"])

    def add_result(self, path):
        item = QListWidgetItem(QIcon(path), Path(path).stem.split("-")[-2])
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self.gallery.addItem(item)
        self.gallery.setCurrentItem(item)

    def select(self, item, *_):
        self.selected = item.data(Qt.ItemDataRole.UserRole) if item else None
        self.selected_metadata = {}
        if self.selected:
            image = QPixmap(self.selected)
            self.preview.setPixmap(
                image.scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            try:
                from PIL import Image

                with Image.open(self.selected) as png:
                    data = json.loads(png.info.get("FaceArt Studio", "{}"))
                self.selected_metadata = data
                self.details.setText(
                    f"Seed {data.get('seed', '—')} · {data.get('size', '—')} px · {data.get('seconds', 0):.1f}s\n{data.get('prompt', '')[:240]}"
                )
            except (ValueError, OSError):
                self.details.setText(Path(self.selected).name)
        self.use_button.setEnabled(bool(self.selected) and not self.busy)
        self.refine_button.setEnabled(bool(self.selected) and not self.busy)
        self.settings_button.setEnabled(bool(self.selected) and not self.busy)
        self.blend_button.setEnabled(
            not self.busy and bool(self.selected_metadata.get("reference_sha256"))
        )

    def capture_reference(self):
        if self.busy:
            return
        index = self.studio.camera.currentData()
        if index is None:
            self.status.setText(
                "Select a camera in Studio first, then reopen Face Lab."
            )
            return
        from modules.studio.reference_camera import ReferenceCamera

        dialog = ReferenceCamera(
            index,
            self.studio.camera.currentText(),
            self.output.parent / "references",
            self,
        )
        if dialog.exec() and dialog.path:
            self.set_reference(dialog.path)
            self.status.setText(
                "Snapshot selected. Describe the facial changes you want."
            )
        dialog.deleteLater()

    def blend_selected(self):
        if self.busy or not self.selected:
            return
        digest = self.selected_metadata.get("reference_sha256")
        if not digest:
            self.status.setText(
                "Generate an image-to-image edit first to blend it with its reference."
            )
            return
        reference = self.studio.settings.get("facegen_references", {}).get(digest)
        try:
            valid = (
                reference
                and Path(reference).is_file()
                and file_hash(reference) == digest
            )
        except OSError:
            valid = False
        if not valid:
            reference, _ = QFileDialog.getOpenFileName(
                self,
                "Locate the original reference used for this edit",
                "",
                "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
            )
            if not reference:
                return
            try:
                matches = file_hash(reference) == digest
            except OSError as error:
                self.status.setText("Cannot read reference photo: " + str(error))
                return
            if not matches:
                self.status.setText(
                    "That is not the original reference used for this edit. Choose the matching photo."
                )
                return
            self.set_reference(reference)
        from modules.studio.snapshot_blend_dialog import SnapshotBlend

        dialog = SnapshotBlend(reference, self.selected, self.output, self)
        if dialog.exec() and dialog.path:
            self.add_result(dialog.path)
            self.status.setText(
                "Blended face saved. Review it, then choose Use this face."
            )
        dialog.deleteLater()

    def refine_selected(self):
        if self.selected and not self.busy:
            self.set_reference(self.selected)
            self.prompt.clear()
            self.prompt.setFocus()

    def reuse_settings(self):
        data = getattr(self, "selected_metadata", {})
        if data and not self.busy:
            self.prompt.setPlainText(data.get("prompt", ""))
            self.seed.setText(str(data.get("seed", "")))
            index = self.size.findData(data.get("size", 768))
            self.size.setCurrentIndex(max(0, index))
            if data.get("mode") == "txt2img":
                self.mode.setCurrentIndex(0)
            else:
                self.mode.setCurrentIndex(1)
                self.status.setText(
                    "Settings restored. Choose the original reference photo to reproduce this edit."
                )

    def use_face(self):
        if self.selected and not self.busy:
            self.studio._set_source(self.selected)
            self.accept()

    def cancel(self):
        self.set_busy(False)
        stop_worker(self.process)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText(
            "Cancelled. Completed candidates are kept; GPU memory released."
        )

    def process_error(self, error):
        if self.busy and error == QProcess.ProcessError.FailedToStart:
            self.set_busy(False)
            self.status.setText(
                "Could not start Face Lab: " + self.process.errorString()
            )

    def finished(self, *_):
        self.read_output()
        self.read_errors()
        if self.busy:
            self.set_busy(False)
            self.progress.setRange(0, 100)
            self.status.setText(
                "Generation worker stopped. "
                + (self.errors[-2500:] or "Try RAM offload and a smaller size.")
            )

    def done(self, result):
        self.cancel()
        super().done(result)
