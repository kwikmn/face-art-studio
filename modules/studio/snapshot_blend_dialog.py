"""Review and feather a local generated edit into its original reference."""

from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QComboBox,
    QLabel,
)
from modules.studio.snapshot_blend import prepare_pair, blend_pair, save_blend


class SnapshotBlend(QDialog):
    def __init__(self, reference, edited, folder, parent=None):
        super().__init__(parent)
        from modules.studio.app import label, button

        self.reference, self.edited, self.folder = reference, edited, folder
        self.path = None
        self.pair = None
        self.result = None
        self.setWindowTitle("Blend with original reference")
        self.resize(920, 800)
        self.setStyleSheet("QDialog {background:#101619;}")
        layout = QVBoxLayout(self)
        layout.addWidget(label("Blend with your snapshot", "title"))
        layout.addWidget(
            label(
                "Original hair, beard and background stay outside the blend. This intentionally preserves parts of the original photo.",
                "muted",
            )
        )
        self.view = QComboBox()
        self.view.addItems(
            ["Blended result", "Original reference", "Aligned edit", "Blend mask"]
        )
        self.view.currentIndexChanged.connect(self.render)
        layout.addWidget(self.view)
        self.preview = QLabel("Aligning the faces…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 320)
        layout.addWidget(self.preview, 1)
        self.controls = {}
        for key, title, low, high, value in [
            ("coverage", "Blend area", 65, 140, 100),
            ("feather", "Edge softness", 5, 50, 25),
            ("offset_x", "Horizontal position", -25, 25, 0),
            ("offset_y", "Vertical position", -25, 25, 0),
        ]:
            row = QHBoxLayout()
            caption = label(f"{title} · {value}%", "muted")
            caption.setMinimumWidth(185)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(low, high)
            slider.setValue(value)
            slider.valueChanged.connect(
                lambda v, c=caption, t=title: c.setText(f"{t} · {v}%")
            )
            slider.valueChanged.connect(self.schedule_render)
            self.controls[key] = slider
            row.addWidget(caption)
            row.addWidget(slider)
            layout.addLayout(row)
        self.status = label("Aligning with facial landmarks…", "status")
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.save_button = button("Save blended face", self.save, "primary")
        self.save_button.setEnabled(False)
        row.addWidget(self.save_button)
        row.addWidget(button("Cancel", self.reject))
        layout.addLayout(row)
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.render)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = self.pool.submit(prepare_pair, reference, edited)
        self.poll = QTimer(self)
        self.poll.timeout.connect(self.ready)
        self.poll.start(100)

    def ready(self):
        if not self.future.done():
            return
        self.poll.stop()
        try:
            self.pair = self.future.result()
            self.render()
            self.save_button.setEnabled(True)
            self.status.setText(
                "Review the blend. Smaller areas preserve more of the original hair and beard."
            )
        except Exception as error:
            self.status.setText(str(error))
            self.preview.setText("Could not align these photos")

    def settings(self):
        return {key: slider.value() / 100 for key, slider in self.controls.items()}

    def schedule_render(self, *_):
        self.render_timer.start(40)

    def render(self, *_):
        if self.pair is None:
            return
        from modules.studio.app import pixmap
        import cv2
        import numpy as np

        self.result, mask = blend_pair(self.pair, **self.settings())
        views = [
            self.result,
            self.pair["original"],
            self.pair["aligned"],
            cv2.cvtColor(np.rint(mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        ]
        self.preview.setPixmap(
            pixmap(views[self.view.currentIndex()]).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def save(self):
        if self.pair is None:
            return
        try:
            self.render()
            self.path = save_blend(
                self.result,
                self.folder,
                {},
                self.reference,
                self.edited,
                self.settings(),
                self.pair["transform"],
            )
            self.accept()
        except Exception as error:
            self.status.setText("Could not save blend: " + str(error))

    def done(self, result):
        self.poll.stop()
        self.render_timer.stop()
        self.future.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().done(result)
