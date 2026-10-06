"""FaceArt Studio: local desktop camera, recording, and postwork workspace."""

from pathlib import Path
from collections import deque
from datetime import datetime
from dataclasses import asdict
import json
import os
import sys
import threading
import time

import cv2
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize, QUrl, QProcess
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QImage,
    QPixmap,
    QIcon,
    QDesktopServices,
    QShortcut,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QColorDialog,
    QSlider,
    QSpinBox,
    QProgressBar,
    QSizePolicy,
    QButtonGroup,
    QScrollArea,
    QSplitter,
    QDialog,
    QGridLayout,
)

from modules import imread_unicode
from modules.live_pipeline import LiveConfig, LivePipeline
from modules.live_face import LiveFaceProcessor
from modules.video_capture import VideoCapturer
from modules.virtual_camera import VirtualCameraOutput
from modules.studio.window_controls import WindowHeader, screen_menu
from modules.studio.recording import Recorder, audio_devices
from modules.studio.background import BackgroundSettings
from modules.studio.postwork import MODELS, export_media

SETTINGS = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    / "FaceArtStudio"
    / "settings.json"
)
STYLE = """
* { font-family: "Segoe UI"; font-size: 13px; color: #e7eeed; }
QMainWindow, QWidget#root { background: #101619; }
QFrame#panel { background: #151d21; border: 1px solid #303c42; border-radius: 12px; }
QFrame#header { background: #151d21; border-bottom: 1px solid #303c42; }
QFrame#section { background: #192226; border: 1px solid #344149; border-radius: 10px; }
QLabel { background: transparent; border: none; }
QLabel#title { font-size: 25px; font-weight: 600; }
QLabel#heading { font-size: 18px; font-weight: 600; }
QLabel#eyebrow { color: #88dfc5; font-size: 11px; font-weight: 600; }
QLabel#muted { color: #9caeb4; font-size: 12px; }
QLabel#status { color: #88dfc5; font-size: 12px; }
QPushButton { background: #202b30; border: 1px solid #43535b; border-radius: 8px; padding: 10px 15px; }
QPushButton:hover { background: #2c3c42; border-color: #88dfc5; }
QPushButton:pressed { background: #364c51; }
QPushButton:checked { background: #25423d; border-color: #88dfc5; color: #abf1dc; }
QPushButton#primary { background: #88dfc5; color: #102d27; border: none; font-weight: 600; }
QPushButton#primary:hover { background: #a6efd9; }
QPushButton#record { border-color: #a45f62; color: #ffb7b9; }
QPushButton#record:checked { background: #6c3036; color: white; }
QPushButton:disabled { color: #63757c; background: #192226; border-color: #2c383e; }
QComboBox { background: #202a30; border: 1px solid #43535b; border-radius: 7px; padding: 9px 10px; min-height: 18px; }
QComboBox QAbstractItemView { background: #202a30; selection-background-color: #33594d; }
QCheckBox { spacing: 9px; padding: 5px 0; }
QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #53666c; border-radius: 5px; background: #202b30; }
QCheckBox::indicator:checked { background: #88dfc5; border-color: #88dfc5; }
QSlider::groove:horizontal { background: #39464d; height: 5px; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #88dfc5; border-radius: 2px; }
QSlider::handle:horizontal { background: #e8fff8; border: 0; width: 15px; margin: -5px 0; border-radius: 7px; }
QProgressBar { background: #263239; border: none; border-radius: 4px; min-height: 7px; color: #c5ddd5; }
QProgressBar::chunk { background: #88dfc5; border-radius: 4px; }
QSplitter::handle:horizontal { background: #303c42; border-radius: 3px; margin: 8px 2px; }
QSplitter::handle:horizontal:hover { background: #88dfc5; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: #182125; width: 7px; }
QScrollBar::handle:vertical { background: #43535b; border-radius: 3px; min-height: 30px; }
QToolTip { background: #25363d; color: white; border: 1px solid #52686f; padding: 6px; }
"""


class SelectableLabel(QLabel):
    """Keep selected status text intact when a timer repeats the same message."""

    def setText(self, text):
        if text != self.text():
            super().setText(text)


def label(text, style=None):
    result = SelectableLabel(text)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse
        | Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    if style:
        result.setObjectName(style)
    result.setWordWrap(True)
    return result


def button(text, callback, kind=None):
    result = QPushButton(text)
    result.setCursor(Qt.CursorShape.PointingHandCursor)
    if kind:
        result.setObjectName(kind)
    result.clicked.connect(callback)
    return result


def panel(kind="panel"):
    widget = QFrame()
    widget.setObjectName(kind)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(14)
    return widget, layout


def pixmap(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(
        QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
    )


class MicrophoneCombo(QComboBox):
    refreshRequested = Signal()

    def __init__(self):
        super().__init__()
        self.popup_pending = False

    def showPopup(self):
        self.popup_pending = True
        self.refreshRequested.emit()

    def refreshed(self):
        if self.popup_pending and self.hasFocus():
            super().showPopup()
        self.popup_pending = False

    def focusOutEvent(self, event):
        self.popup_pending = False
        super().focusOutEvent(event)


class Preview(QWidget):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.message = "A new perspective. Still you."
        self.detail = "Choose a face and start your camera."

    def set_frame(self, frame):
        self.frame = pixmap(frame) if frame is not None else None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#0d1316"))
        painter.setPen(QPen(QColor("#344149"), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 10, 10)
        if self.frame:
            size = self.frame.size().scaled(
                self.size() - QSize(4, 4), Qt.AspectRatioMode.KeepAspectRatio
            )
            x, y = (
                (self.width() - size.width()) // 2,
                (self.height() - size.height()) // 2,
            )
            painter.drawPixmap(x, y, size.width(), size.height(), self.frame)
        else:
            painter.setPen(QColor("#8fe1c8"))
            font = painter.font()
            font.setPointSize(20)
            painter.setFont(font)
            painter.drawText(
                self.rect().adjusted(15, -25, -15, -25),
                Qt.AlignmentFlag.AlignCenter,
                self.message,
            )
            font.setPointSize(11)
            painter.setFont(font)
            painter.setPen(QColor("#82969e"))
            painter.drawText(
                self.rect().adjusted(15, 40, -15, 40),
                Qt.AlignmentFlag.AlignCenter,
                self.detail,
            )


class Sparkline(QWidget):
    def __init__(self):
        super().__init__()
        self.values = deque(maxlen=90)
        self.setFixedHeight(30)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#88dfc5"), 1.5))
        values = list(self.values)
        for i in range(1, len(values)):
            p.drawLine(
                round((i - 1) * self.width() / 90),
                round(self.height() - min(30, values[i - 1]) / 30 * 26),
                round(i * self.width() / 90),
                round(self.height() - min(30, values[i]) / 30 * 26),
            )


class Events(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)


class Studio(QMainWindow):
    def __init__(self, source=None, enumerate_devices=True):
        super().__init__()
        self.setWindowTitle("FaceArt Studio v0.2 experimental")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, screen.width() - 40), min(940, screen.height() - 40))
        self.setMinimumSize(1080, min(760, screen.height() - 60))
        self.session = None
        self.recorder = None
        self.projection = None
        self.tasks = []
        self.job_cancel = threading.Event()
        self.job_busy = False
        self.pending_close = False
        self.pending_stop = False
        self.record_saving = False
        self.mode = "Live camera"
        self.media_path = None
        self.last_export = None
        self.last_metrics = 0
        self.last_preview = 0
        self.sequence = -1
        self.last_record_error = None
        self.last_source_status = None
        self.audio_refresh_busy = False
        self.live_audio = None
        self.live_audio_error = None
        self.live_audio_stopping = False
        self.live_restorer = None
        self.live_load_busy = False
        try:
            self.settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.settings = {}
        try:
            self.background_config = BackgroundSettings(**self.settings.get("background", {}))
        except (TypeError, ValueError):
            self.background_config = BackgroundSettings()
        from modules.studio.slideshow import Slideshow
        self.slideshow = Slideshow(self.settings.get('slideshow',{}))
        self.slideshow_dialog = None
        self.source = source or self.settings.get("source")
        self.library = [
            p for p in self.settings.get("library", []) if Path(p).is_file()
        ]
        self.output_folder = Path(
            self.settings.get("output", str(Path.home() / "Videos" / "FaceArt Studio"))
        )
        self._build()
        for key,callback in [('Ctrl+Alt+Left',lambda:self.slide_navigate(-1)),('Ctrl+Alt+Right',lambda:self.slide_navigate(1)),('Ctrl+Alt+Space',self.slide_toggle)]:
            shortcut=QShortcut(QKeySequence(key),self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
        self.microphone.activated.connect(lambda *_: self._save())
        self._set_source(self.source, save=False)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(67)
        self.shortcut = QShortcut(QKeySequence("Ctrl+Shift+P"), self)
        self.shortcut.activated.connect(self.pause_video)
        QApplication.instance().aboutToQuit.connect(self._shutdown)
        if enumerate_devices:
            QTimer.singleShot(100, self.refresh_devices)

    def _task(self, fn, done, progress=None, failed=None):
        signals = Events(self)
        signals.done.connect(done)
        signals.failed.connect(failed or self._task_error)
        if progress:
            signals.progress.connect(progress)

        def run():
            try:
                signals.done.emit(fn(signals))
            except Exception as exc:
                signals.failed.emit(str(exc))

        thread = threading.Thread(target=run, daemon=True)
        self.tasks.append((thread, signals))
        thread.start()

    def _task_error(self, message):
        self.notice.setText(message)
        self.job_busy = False
        self.record_saving = False
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self.pending_close:
            self.close()

    def _build(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 0, 12, 8)
        layout.setSpacing(12)
        header = WindowHeader()
        h = QHBoxLayout(header)
        h.setContentsMargins(12, 18, 12, 18)
        brand = label("FaceArt Studio", "title")
        brand.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        h.addWidget(brand)
        tagline = label("YOUR FACE. YOUR EXPRESSION.", "eyebrow")
        tagline.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        h.addWidget(tagline)
        h.addStretch()
        self.tabs = QButtonGroup(self)
        self.tabs.setExclusive(True)
        for text in ("Live camera", "Image", "Video"):
            b = button(text, lambda checked=False, t=text: self.set_mode(t))
            b.setCheckable(True)
            b.setChecked(text == "Live camera")
            self.tabs.addButton(b)
            h.addWidget(b)
        h.addStretch()
        h.addWidget(button("Face lab", self.open_face_lab))
        self.screen_button = button(
            "Move to screen", lambda: screen_menu(self, self.screen_button)
        )
        h.addWidget(self.screen_button)
        layout.addWidget(header)
        self.body_splitter = QSplitter(Qt.Orientation.Horizontal)
        body = self.body_splitter
        body.setChildrenCollapsible(False)
        body.setHandleWidth(10)
        layout.addWidget(body, 1)
        left, l = panel()
        left.setMinimumWidth(270)
        l.setSpacing(8)
        l.addWidget(label("Source face", "heading"))
        self.source_preview = QLabel("Choose your face")
        self.source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_preview.setFixedSize(228, 180)
        self.source_preview.setStyleSheet(
            "background:#202b30;border-radius:9px;color:#8ba2ab;"
        )
        l.addWidget(self.source_preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        l.addWidget(button("Change face", self.choose_face, "primary"))
        saved_row = QHBoxLayout()
        saved_row.addWidget(label("SAVED FACES", "eyebrow"))
        saved_row.addStretch()
        saved_row.addWidget(button("Library", self.show_library))
        l.addLayout(saved_row)
        self.library_row = QHBoxLayout()
        l.addLayout(self.library_row)
        l.addWidget(label("Input", "heading"))
        self.camera = QComboBox()
        self.camera.addItem("Finding cameras…", None)
        l.addWidget(self.camera)
        row = QHBoxLayout()
        self.resolution = QComboBox()
        self.resolution.addItem("1280 × 720", 720)
        self.resolution.addItem("960 × 540", 540)
        row.addWidget(self.resolution, 1)
        refresh = button("Refresh", self.refresh_devices)
        refresh.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        row.addWidget(refresh)
        l.addLayout(row)
        self.mirror = QCheckBox("Mirror video")
        self.mirror.setChecked(self.settings.get("mirror", False))
        l.addWidget(self.mirror)
        l.addWidget(label("Audio", "heading"))
        self.microphone = MicrophoneCombo()
        self.microphone.refreshRequested.connect(self.refresh_microphones)
        self.microphone.addItem("Finding microphones…", None)
        l.addWidget(self.microphone)
        self.audio_hint = label(
            "Choose any microphone or virtual audio input.", "muted"
        )
        l.addWidget(self.audio_hint)
        delay_row = QHBoxLayout()
        delay_row.addWidget(label("Audio delay", "muted"))
        self.audio_delay = QSpinBox()
        self.audio_delay.setRange(0, 2000)
        self.audio_delay.setSingleStep(10)
        self.audio_delay.setSuffix(" ms")
        self.audio_delay.setValue(self.settings.get("audio_delay_ms", 0))
        self.audio_delay.setStyleSheet("QSpinBox { background:#192226; color:#e7eeed; border:1px solid #344149; padding:5px; }")
        delay_row.addWidget(self.audio_delay)
        l.addLayout(delay_row)
        self.audio_delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.audio_delay_slider.setRange(0, 2000)
        self.audio_delay_slider.setSingleStep(10)
        self.audio_delay_slider.setPageStep(100)
        self.audio_delay_slider.setValue(self.audio_delay.value())
        self.audio_delay_slider.valueChanged.connect(self.audio_delay.setValue)
        self.audio_delay.valueChanged.connect(self.audio_delay_slider.setValue)
        self.audio_delay.valueChanged.connect(lambda *_: self._save())
        hint = "Positive delay plays audio later in recordings and streaming audio. Stop both before changing it."
        self.audio_delay.setToolTip(hint)
        self.audio_delay_slider.setToolTip(hint)
        l.addWidget(self.audio_delay_slider)
        self.audio_output = MicrophoneCombo()
        self.audio_output.refreshRequested.connect(self.refresh_audio_outputs)
        self.audio_output.activated.connect(lambda *_: self._save())
        l.addWidget(self.audio_output)
        self.stream_audio_button = QPushButton("Start streaming audio")
        self.stream_audio_button.clicked.connect(self.toggle_stream_audio)
        l.addWidget(self.stream_audio_button)
        self.stream_audio_hint = label("Choose a virtual cable, then select its recording endpoint in your streaming app.", "muted")
        self.stream_audio_hint.setWordWrap(True)
        l.addWidget(self.stream_audio_hint)
        self.refresh_audio_outputs()
        l.addWidget(label("Output", "heading"))
        self.virtual = QPushButton("Start virtual camera")
        self.virtual.setCheckable(True)
        self.virtual.setChecked(False)
        self.virtual.setEnabled(False)
        self.virtual.toggled.connect(self.toggle_virtual)
        l.addWidget(self.virtual)
        self.virtual.setToolTip(
            "Explicit output to the installed OBS Virtual Camera. Apps already using that device may receive this video. Select it separately in your receiving app. Do not start OBS's own virtual-camera output at the same time. Audio stays separate."
        )
        self.virtual_status=label('Off · Start camera first. Audio remains in Voicemod.','muted')
        l.addWidget(self.virtual_status)
        l.addWidget(button("Open projection", self.open_projection))
        l.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(278)
        body.addWidget(left_scroll)
        center, c = panel()
        heading = QHBoxLayout()
        self.preview_title = label("Live preview", "heading")
        heading.addWidget(self.preview_title)
        heading.addStretch()
        self.camera_status = label("●  Camera stopped", "status")
        heading.addWidget(self.camera_status)
        c.addLayout(heading)
        self.preview = Preview()
        c.addWidget(self.preview, 1)
        meters, m = panel("section")
        m.setContentsMargins(12, 8, 12, 8)
        metrics = QHBoxLayout()
        self.fps_label = label("— FPS", "status")
        metrics.addWidget(self.fps_label)
        self.timing = label("Ready when you are", "muted")
        metrics.addWidget(self.timing)
        metrics.addStretch()
        self.spark = Sparkline()
        self.spark.setMinimumWidth(150)
        metrics.addWidget(self.spark)
        m.addLayout(metrics)
        c.addWidget(meters)
        controls = QHBoxLayout()
        self.start_button = button("Start camera", self.toggle_camera, "primary")
        controls.addWidget(self.start_button, 2)
        self.pause_button = button("Pause", self.pause_video)
        self.pause_button.setEnabled(False)
        controls.addWidget(self.pause_button)
        self.capture_button = button("Capture frame", self.capture_frame)
        self.capture_button.setEnabled(False)
        controls.addWidget(self.capture_button)
        self.record_button = button("●  Record", self.toggle_record, "record")
        self.record_button.setEnabled(False)
        controls.addWidget(self.record_button)
        c.addLayout(controls)
        file_controls = QHBoxLayout()
        self.import_button = button("Open image / video", self.choose_media)
        file_controls.addWidget(self.import_button)
        self.show_output = button("Open last export", self.open_last)
        self.show_output.setEnabled(False)
        file_controls.addWidget(self.show_output)
        self.folder_button = button("Recordings folder ↗", self.open_folder)
        file_controls.addWidget(self.folder_button)
        c.addLayout(file_controls)
        self.notice = label(
            "Protected output only. Your original camera image is never used as a fallback.",
            "muted",
        )
        c.addWidget(self.notice)
        body.addWidget(center)
        right, r = panel()
        right.setMinimumWidth(290)
        r.addWidget(label("Finish your look", "heading"))
        card, a = panel("section")
        a.addWidget(label("LIVE OUTPUT", "eyebrow"))
        a.addWidget(label("Full face replacement", "heading"))
        a.addWidget(
            label(
                "Face changes apply while the camera runs. Brief detection gaps repeat the last processed frame.",
                "muted",
            )
        )
        self.preview_enabled = QCheckBox("Show live preview")
        self.preview_enabled.setChecked(True)
        a.addWidget(self.preview_enabled)
        self.live_gpen = QCheckBox("Live GPEN 256")
        self.live_gpen.setChecked(self.settings.get("live_gpen", False))
        self.live_gpen.toggled.connect(self.configure_live_gpen)
        a.addWidget(self.live_gpen)
        self.live_strength = QSlider(Qt.Orientation.Horizontal)
        self.live_strength.setRange(0, 100)
        self.live_strength.setValue(self.settings.get("live_strength", 60))
        self.live_strength.valueChanged.connect(self.configure_live_gpen)
        a.addWidget(self.live_strength)
        self.live_gpen_status = label("Optional live restoration · 60%", "muted")
        a.addWidget(self.live_gpen_status)
        self.jaw_label = label("Jaw feathering", "muted")
        a.addWidget(self.jaw_label)
        self.jaw_feather = QSlider(Qt.Orientation.Horizontal)
        self.jaw_feather.setRange(0, 100)
        self.jaw_feather.setValue(self.settings.get("jaw_feather", 60))
        self.jaw_feather.setToolTip("Softens the lower face transition. Central facial features stay fully replaced.")
        self.jaw_label.setText(f"Jaw feathering · {self.jaw_feather.value()}%")
        self.jaw_feather.valueChanged.connect(self.configure_jaw_feather)
        a.addWidget(self.jaw_feather)
        a.addWidget(label("Up to 20 fresh FPS · 30 FPS output", "status"))
        r.addWidget(card)
        card, a = panel("section")
        a.addWidget(label("BACKGROUND", "eyebrow"))
        self.background_backend = QComboBox()
        for title, value in (("MediaPipe — fallback", "mediapipe"), ("RVM — recurrent matting", "rvm"), ("MODNet — portrait matting", "modnet")):
            self.background_backend.addItem(title, value)
        self.background_backend.setCurrentIndex(self.background_backend.findData(self.background_config.backend))
        a.addWidget(self.background_backend)
        self.background_provider = QComboBox()
        self.background_provider.addItem("CPU", "cpu")
        self.background_provider.addItem("CUDA", "cuda")
        self.background_provider.setCurrentIndex(self.background_provider.findData(self.background_config.provider))
        a.addWidget(self.background_provider)
        a.addWidget(button("Open offline Matting Lab", self.open_matting_lab))
        self.background_mode = QComboBox()
        for title, mode in (("Original background", "off"), ("Blur background", "blur"),
                            ("Solid color", "color"), ("Background image", "image"), ("Live slideshow", "slideshow")):
            self.background_mode.addItem(title, mode)
        self.background_mode.setCurrentIndex(self.background_mode.findData(self.background_config.mode))
        a.addWidget(self.background_mode)
        self.background_color = button("Choose color…", self.choose_background_color)
        a.addWidget(self.background_color)
        self.background_image = button("Choose image…", self.choose_background_image)
        a.addWidget(self.background_image)
        a.addWidget(button('Open background slideshow…',self.open_slideshow))
        self.background_image_name = label("", "muted")
        a.addWidget(self.background_image_name)
        self.background_blur = QSlider(Qt.Orientation.Horizontal)
        self.background_blur.setRange(1, 50)
        self.background_blur.setValue(self.background_config.blur)
        a.addWidget(label("Blur strength", "muted"))
        a.addWidget(self.background_blur)
        self.background_softness = QSlider(Qt.Orientation.Horizontal)
        self.background_softness.setRange(0, 20)
        self.background_softness.setValue(self.background_config.softness)
        a.addWidget(label("Edge softness", "muted"))
        a.addWidget(self.background_softness)
        self.background_smoothing = QSlider(Qt.Orientation.Horizontal)
        self.background_smoothing.setRange(0, 60)
        self.background_smoothing.setValue(self.background_config.smoothing)
        self.background_smoothing.setToolTip("Reduce edge flicker. Higher values may trail behind fast movement.")
        a.addWidget(label("Motion smoothing", "muted"))
        a.addWidget(self.background_smoothing)
        self.background_chair = QSlider(Qt.Orientation.Horizontal)
        self.background_chair.setRange(0, 100)
        self.background_chair.setValue(self.background_config.chair_cleanup)
        self.background_chair.setToolTip("Use the detected face and a soft shoulder estimate to reject uncertain chair pixels. Reduce if hair or raised arms disappear.")
        self.background_chair_label = label("", "muted")
        a.addWidget(self.background_chair_label)
        a.addWidget(self.background_chair)
        self.background_tightness = QSlider(Qt.Orientation.Horizontal)
        self.background_tightness.setRange(0, 100)
        self.background_tightness.setValue(self.background_config.tightness)
        self.background_tightness.setToolTip("Remove more uncertain foreground and translucent edges. High values may trim fine hair.")
        self.background_tightness_label = label("", "muted")
        a.addWidget(self.background_tightness_label)
        a.addWidget(self.background_tightness)
        self.background_status = label("", "muted")
        a.addWidget(self.background_status)
        self._update_background_controls()
        self.background_mode.currentIndexChanged.connect(self.configure_background)
        self.background_backend.currentIndexChanged.connect(self.configure_background)
        self.background_provider.currentIndexChanged.connect(self.configure_background)
        for slider in (self.background_blur, self.background_softness, self.background_smoothing, self.background_chair, self.background_tightness):
            slider.valueChanged.connect(self.configure_background)
        r.addWidget(card)
        card, a = panel("section")
        a.addWidget(label("AI POSTWORK", "eyebrow"))
        a.addWidget(label("Restore the details", "heading"))
        a.addWidget(
            label(
                "Clean up a captured image or recording after the camera stops. Export a new copy.",
                "muted",
            )
        )
        self.model = QComboBox()
        self.model.addItems(MODELS)
        a.addWidget(self.model)
        strength_row = QHBoxLayout()
        strength_row.addWidget(label("Restoration strength"))
        self.strength_label = label("60%", "status")
        strength_row.addWidget(self.strength_label)
        a.addLayout(strength_row)
        self.strength = QSlider(Qt.Orientation.Horizontal)
        self.strength.setRange(0, 100)
        self.strength.setValue(60)
        self.strength.valueChanged.connect(
            lambda v: self.strength_label.setText(f"{v}%")
        )
        a.addWidget(self.strength)
        self.apply_swap = QCheckBox("Replace face")
        self.apply_swap.setToolTip(
            "Apply the selected source face to each frame of imported media"
        )
        a.addWidget(self.apply_swap)
        self.apply_enhance = QCheckBox("Apply AI restoration")
        self.apply_enhance.setChecked(True)
        a.addWidget(self.apply_enhance)
        self.export_button = button("Enhance / export…", self.export_job, "primary")
        a.addWidget(self.export_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        a.addWidget(self.progress)
        self.cancel_button = button("Cancel export", self.cancel_export)
        self.cancel_button.setEnabled(False)
        a.addWidget(self.cancel_button)
        a.addWidget(
            label(
                "Models download on first use. Video exports retain source audio.",
                "muted",
            )
        )
        r.addWidget(card)
        r.addWidget(button("Expression controls · planned", self.show_expression_plan))
        r.addStretch()
        r.addWidget(button("Export location…", self.choose_folder))
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setWidget(right)
        right_scroll.setMinimumWidth(298)
        body.addWidget(right_scroll)
        for index in range(3):
            body.setStretchFactor(index, 1 if index == 1 else 0)
        sizes = self.settings.get("panel_sizes", [310, 800, 330])
        if not (isinstance(sizes, list) and len(sizes) == 3
                and all(type(size) is int and size > 0 for size in sizes)):
            sizes = [310, 800, 330]
        body.setSizes(sizes)
        self.panel_save_timer = QTimer(self)
        self.panel_save_timer.setSingleShot(True)
        self.panel_save_timer.setInterval(300)
        self.panel_save_timer.timeout.connect(self._save)
        body.splitterMoved.connect(lambda *_: self.panel_save_timer.start())
        for index in (1, 2):
            body.handle(index).setToolTip("Drag to resize the sidebar")
        # Device names should use the available width rather than force it wider.
        for combo in (self.camera, self.resolution, self.microphone, self.audio_output, self.model):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Keep controls intact at the narrow end, including scrollbar space.
        left_scroll.setMinimumWidth(max(278, left.minimumSizeHint().width() + 8))
        right_scroll.setMinimumWidth(max(298, right.minimumSizeHint().width() + 8))
        footer = QHBoxLayout()
        footer.addWidget(label("●  LOCAL PROCESSING", "eyebrow"))
        footer.addWidget(
            label("GPU inference  ·  NVENC recording  ·  No cloud upload", "muted")
        )
        footer.addStretch()
        self.footer = label("FaceArt Studio · Preview build", "muted")
        footer.addWidget(self.footer)
        layout.addLayout(footer)
        self._library_refresh()

    def _save(self):
        self.settings.update(
            source=self.source,
            library=self.library[:12],
            output=str(self.output_folder),
            mirror=self.mirror.isChecked(),
        )
        if self.audio_output.currentData():
            self.settings["stream_audio_output"] = self.audio_output.currentData()
        if self.microphone.currentData() is not None:
            self.settings["microphone"] = self.microphone.currentData()
        self.settings.update(
            live_gpen=self.live_gpen.isChecked(),
            live_strength=self.live_strength.value(),
            jaw_feather=self.jaw_feather.value(),
            audio_delay_ms=self.audio_delay.value(),
            background=asdict(self.background_config),
            slideshow=self.slideshow.settings(),
        )
        if self.isVisible():
            self.settings["panel_sizes"] = self.body_splitter.sizes()
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        temporary = SETTINGS.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        temporary.replace(SETTINGS)

    def _set_source(self, path, save=True):
        if not path or not Path(path).is_file():
            return
        image = imread_unicode(str(path))
        if image is None:
            self.notice.setText("Cannot read that face image")
            return
        self.source = str(path)
        self.source_preview.setPixmap(
            pixmap(image).scaled(
                228,
                180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.library = [self.source] + [p for p in self.library if p != self.source]
        self._library_refresh()
        if self.session:
            self.session.change_source(self.source)
        if save:
            self._save()

    def _library_refresh(self):
        while self.library_row.count():
            item = self.library_row.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        for path in self.library[:3]:
            b = button("", lambda checked=False, p=path: self._set_source(p))
            b.setFixedSize(54, 54)
            image = imread_unicode(path)
            if image is not None:
                b.setIcon(QIcon(pixmap(image)))
                b.setIconSize(QSize(46, 46))
            b.setToolTip(Path(path).name)
            self.library_row.addWidget(b)
        self.library_row.addStretch()

    def show_library(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Face library")
        dialog.setStyleSheet("QDialog { background: #151d21; }")
        grid = QGridLayout(dialog)
        if not self.library:
            grid.addWidget(label("Choose a face image to start your library."), 0, 0)
        for i, path in enumerate(self.library[:12]):

            def choose(checked=False, selected=path):
                self._set_source(selected)
                dialog.accept()

            entry = button(Path(path).stem, choose)
            entry.setFixedSize(160, 180)
            image = imread_unicode(path)
            if image is not None:
                entry.setIcon(QIcon(pixmap(image)))
                entry.setIconSize(QSize(110, 130))
            entry.setToolTip(path)
            grid.addWidget(entry, i // 4, i % 4)
        dialog.exec()

    def choose_face(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a source face",
            str(Path(self.source).parent) if self.source else "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            self._set_source(path)

    def refresh_microphones(self):
        if self.recorder or self.live_audio:
            self.microphone.refreshed()
            return
        if self.audio_refresh_busy:
            return
        self.audio_refresh_busy = True
        self.audio_hint.setText("Refreshing microphones…")
        self._task(
            lambda _: audio_devices(), self._audio_ready, failed=self._audio_failed
        )

    def _audio_failed(self, error):
        self.audio_refresh_busy = False
        self.audio_hint.setText("Could not refresh microphones: " + error)
        self.microphone.refreshed()

    def refresh_devices(self):
        self.refresh_microphones()
        if self.session or self.job_busy:
            return

        def get(signals):
            from pygrabber.dshow_graph import FilterGraph
            import comtypes

            comtypes.CoInitialize()
            try:
                return FilterGraph().get_input_devices()
            finally:
                comtypes.CoUninitialize()

        self._task(get, self._cameras_ready)

    def _devices_ready(self, devices):
        cameras, audio = devices
        self._cameras_ready(cameras)
        self._audio_ready(audio)

    def _cameras_ready(self, cameras):
        if self.session:
            return
        previous = self.camera.currentData()
        self.camera.clear()
        for index, name in enumerate(cameras):
            if not any(v in name.lower() for v in ("virtual", "unity video")):
                self.camera.addItem(name, index)
        if not self.camera.count():
            self.camera.addItem("No physical camera found", None)
        selected = self.camera.findData(previous)
        if selected >= 0:
            self.camera.setCurrentIndex(selected)

    def _audio_ready(self, audio):
        self.audio_refresh_busy = False
        if self.recorder or self.live_audio:
            self.microphone.refreshed()
            return
        previous = self.microphone.currentData()
        if previous is None:
            previous = self.settings.get("microphone")
        self.microphone.clear()
        self.microphone.addItem("Choose microphone…", None)
        for name in audio:
            self.microphone.addItem(name, name)
        self.microphone.addItem("Video only (no audio)", "")
        selected = self.microphone.findData(previous) if previous is not None else -1
        if selected >= 0:
            self.microphone.setCurrentIndex(selected)
        self.audio_hint.setText(
            "List refreshes when opened. Your selected input is preserved."
        )
        self.microphone.refreshed()

    def configure_jaw_feather(self, *_):
        value = self.jaw_feather.value()
        self.jaw_label.setText(f"Jaw feathering · {value}%")
        if self.session:
            self.session.processor.set_jaw_feathering(value / 100)
        self._save()

    def _update_background_controls(self):
        mode = self.background_config.mode
        self.background_color.setEnabled(mode == "color")
        self.background_color.setText(f"Choose color… {self.background_config.color}")
        self.background_image.setEnabled(mode == "image")
        self.background_image_name.setText(Path(self.background_config.image).name if self.background_config.image else "No background image selected")
        self.background_image_name.setVisible(mode == "image")
        self.background_blur.setEnabled(mode == "blur")
        refine = mode != "off" and self.background_config.backend == 'mediapipe'
        self.background_softness.setEnabled(refine)
        self.background_smoothing.setEnabled(mode != "off")
        self.background_chair.setEnabled(refine)
        self.background_tightness.setEnabled(refine)
        self.background_chair_label.setText(f"Chair suppression · {self.background_config.chair_cleanup}%")
        self.background_tightness_label.setText(f"Mask tightness · {self.background_config.tightness}%")
        self.background_status.setText("Original background" if mode == "off" else "Ready · applies to preview, webcam and recordings")

    def configure_background(self, *_):
        mode = self.background_mode.currentData()
        if mode!='slideshow':
            self.slideshow.play(False)
        image = self.background_config.image
        if mode == "image" and not image:
            image, _ = QFileDialog.getOpenFileName(self, "Choose background image", str(Path.home()), "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
            if not image:
                self.background_mode.setCurrentIndex(self.background_mode.findData(self.background_config.mode))
                return
        self.background_config = BackgroundSettings(
            mode=mode, color=self.background_config.color, image=image,
            blur=self.background_blur.value(), softness=self.background_softness.value(),
            smoothing=self.background_smoothing.value(),
            chair_cleanup=self.background_chair.value(), tightness=self.background_tightness.value(),
            backend=self.background_backend.currentData(), provider=self.background_provider.currentData(),
            downsample_ratio=self.background_config.downsample_ratio,
            preserve_alpha=True,
        )
        if self.session:
            self.session.processor.set_background(self.background_config)
        self._update_background_controls()
        self._save()

    def open_matting_lab(self):
        import subprocess
        from modules.paths import ROOT_DIR
        subprocess.Popen([sys.executable, '-s', str(Path(ROOT_DIR) / 'run_matting_lab.py')], cwd=ROOT_DIR)

    def activate_slideshow(self):
        self.background_mode.setCurrentIndex(self.background_mode.findData('slideshow'))

    def open_slideshow(self):
        if self.slideshow_dialog is None:
            from modules.studio.slideshow_ui import SlideshowDialog
            self.slideshow_dialog=SlideshowDialog(self.slideshow,self)
            self.slideshow_dialog.changed.connect(self._save)
            self.slideshow_dialog.activate.connect(self.activate_slideshow)
        self.slideshow_dialog.show();self.slideshow_dialog.raise_()

    def slide_navigate(self,step):
        self.slideshow.navigate(step);self.activate_slideshow();self._save()

    def slide_toggle(self):
        self.slideshow.play(not self.slideshow.playing);self.activate_slideshow();self._save()

    def choose_background_color(self):
        color = QColorDialog.getColor(QColor(self.background_config.color), self, "Background color")
        if color.isValid():
            self.background_config = BackgroundSettings(**{**asdict(self.background_config), "color": color.name()})
            self.configure_background()

    def choose_background_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose background image", str(Path.home()), "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.background_config = BackgroundSettings(**{**asdict(self.background_config), "image": path})
            self.configure_background()

    def configure_live_gpen(self, *_):
        strength = self.live_strength.value() / 100
        enabled = self.live_gpen.isChecked()
        self._save()
        if not enabled:
            if self.session:
                self.session.processor.set_live_enhancement()
            self.live_gpen_status.setText("Live restoration off")
            return
        if not self.session:
            self.live_gpen_status.setText(f"GPEN 256 ready to enable · {strength:.0%}")
            return
        if self.live_restorer is not None:
            self.session.processor.set_live_enhancement(self.live_restorer, strength)
            self.live_gpen_status.setText(f"GPEN 256 active · {strength:.0%}")
        elif not self.live_load_busy:
            self.live_load_busy = True
            self.live_gpen_status.setText("Loading GPEN 256; face swapping continues…")
            from modules.studio.live_enhancement import LiveGPEN

            self._task(
                lambda _: LiveGPEN(),
                self._live_gpen_ready,
                failed=self._live_gpen_failed,
            )

    def _live_gpen_ready(self, enhancer):
        self.live_load_busy = False
        self.live_restorer = enhancer
        self.configure_live_gpen()

    def _live_gpen_failed(self, error):
        self.live_load_busy = False
        self.live_gpen.setChecked(False)
        self.live_gpen_status.setText("Live restoration unavailable: " + error)

    def toggle_virtual(self, enabled):
        if self.session:
            self.session.enable_output(enabled)
            self.virtual.setText('Stop virtual camera' if enabled else 'Start virtual camera')
            self.virtual_status.setText('Starting OBS Virtual Camera…' if enabled else 'Stopping local virtual output…')
        elif enabled:
            self.virtual.blockSignals(True);self.virtual.setChecked(False);self.virtual.blockSignals(False)
            self.virtual_status.setText('Start camera before starting virtual output.')

    def set_mode(self, mode):
        if self.session or self.recorder:
            self.notice.setText("Stop the camera before changing workspaces.")
            for tab in self.tabs.buttons():
                tab.setChecked(tab.text() == self.mode)
            return
        self.mode = mode
        self.import_button.setText(
            "Open " + ("media" if mode == "Live camera" else mode.lower())
        )
        self.preview_title.setText(
            "Live preview" if mode == "Live camera" else f"{mode} workspace"
        )
        if mode != "Live camera":
            self.notice.setText(
                "Open media, then choose restoration and export settings. Stop the camera before postwork."
            )
        elif not self.session:
            self.preview.set_frame(None)

    def toggle_camera(self):
        if self.session:
            if self.recorder:
                self.pending_stop = True
                self.stop_record()
                return
            self.stop_camera()
            return
        if self.job_busy:
            self.notice.setText("Wait for postwork to finish or cancel it first.")
            return
        if not self.source:
            self.choose_face()
            if not self.source:
                return
        index = self.camera.currentData()
        if index is None:
            self.notice.setText("Connect a physical camera and refresh inputs.")
            return
        height = self.resolution.currentData()
        width = 1280 if height == 720 else 960
        try:
            self.session = LivePipeline(
                VideoCapturer(index),
                LiveFaceProcessor(
                    self.source,
                    width=width,
                    height=height,
                    mirror=self.mirror.isChecked(),
                ),
                VirtualCameraOutput(),
                LiveConfig(width, height),
            )
            self.session.processor.set_jaw_feathering(self.jaw_feather.value() / 100)
            self.session.enable_output(self.virtual.isChecked())
            self.session.processor.set_background(self.background_config)
            self.session.processor.slideshow=self.slideshow
            self.slideshow.set_size(width,height)
            self.session.start()
            self.configure_live_gpen()
        except Exception as exc:
            if self.session:
                self.session.stop()
            self.session = None
            self.notice.setText(str(exc))
            return
        self.start_button.setText("Stop camera")
        self.virtual.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.capture_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.camera.setEnabled(False)
        self.resolution.setEnabled(False)
        self.mirror.setEnabled(False)
        self.notice.setText("Loading face models…")
        self._save()

    def stop_camera(self):
        if self.session:
            self.session.stop()
            self.session = None
        self.virtual.blockSignals(True);self.virtual.setChecked(False);self.virtual.blockSignals(False)
        self.virtual.setEnabled(False);self.virtual.setText('Start virtual camera')
        self.virtual_status.setText('Off · camera stopped; virtual device released.')
        self.start_button.setText("Start camera")
        self.pause_button.setText("Pause")
        self.pause_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.camera.setEnabled(True)
        self.resolution.setEnabled(True)
        self.mirror.setEnabled(True)
        self.camera_status.setText("●  Camera stopped")
        self.preview.set_frame(None)
        if self.projection:
            self.projection.set_frame(None)
        self.fps_label.setText("— FPS")
        self.timing.setText("Webcam released")
        if self.notice.text() == "Loading face models.":
            self.notice.setText("Camera stopped.")

    def pause_video(self):
        if self.session:
            paused = not self.session.metrics()["paused"]
            self.session.pause(paused)
            self.pause_button.setText("Resume" if paused else "Pause")

    def _destination(self, suffix, stem="FaceArt"):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        return self.output_folder / f"{stem}-{datetime.now():%Y%m%d-%H%M%S-%f}{suffix}"

    def capture_frame(self):
        if not self.session:
            return
        _, frame = self.session.preview()
        if frame is None:
            return
        path = self._destination(".png")
        ok, data = cv2.imencode(".png", frame)
        if ok:
            data.tofile(str(path))
            self._export_ready(path)

    def refresh_audio_outputs(self):
        from PySide6.QtMultimedia import QMediaDevices

        if self.live_audio:
            self.audio_output.refreshed()
            return
        previous = self.audio_output.currentData() or self.settings.get("stream_audio_output")
        self.audio_output.clear()
        self.audio_output.addItem("Choose virtual audio cable…", None)
        for device in QMediaDevices.audioOutputs():
            if "cable" in device.description().lower():
                self.audio_output.addItem(device.description(), bytes(device.id()).hex())
        index = self.audio_output.findData(previous)
        if index >= 0:
            self.audio_output.setCurrentIndex(index)
        self.stream_audio_hint.setText(
            "Select the cable's recording endpoint (e.g. CABLE Output) in OBS/Discord."
            if self.audio_output.count() > 1 else
            "No virtual audio cable installed. Install VB-CABLE, then open this list to refresh."
        )
        self.audio_output.refreshed()

    def toggle_stream_audio(self):
        if self.live_audio:
            from modules.studio.process_utils import stop_worker
            self.live_audio_stopping = True
            stop_worker(self.live_audio)
            return
        microphone = self.microphone.currentData()
        output = self.audio_output.currentData()
        if not microphone or not output:
            self.stream_audio_hint.setText("Select a microphone and a dedicated virtual audio cable first.")
            return
        if "cable" in microphone.lower():
            self.stream_audio_hint.setText("Select your microphone or voice changer as input, not the cable return.")
            return
        self._save()
        self.live_audio_error = None
        self.live_audio_stopping = False
        process = QProcess(self)
        self.live_audio = process
        process.readyReadStandardOutput.connect(self._stream_audio_messages)
        process.finished.connect(self._stream_audio_finished)
        process.errorOccurred.connect(self._stream_audio_process_error)
        self.audio_output.setEnabled(False)
        self._set_record_audio_enabled(False)
        self.stream_audio_button.setText("Stop streaming audio")
        self.stream_audio_hint.setText("Opening microphone and virtual cable…")
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        process.start(sys.executable, ["-m", "modules.studio.live_audio_worker", json.dumps({
            "microphone": microphone, "output": output, "delay_ms": self.audio_delay.value()
        })])

    def _stream_audio_messages(self):
        process = self.live_audio
        if not process:
            return
        while process.canReadLine():
            line = bytes(process.readLine()).decode("utf-8", errors="replace").strip()
            if not line.startswith("AUDIO:"):
                continue
            try:
                message = json.loads(line[6:])
            except ValueError:
                continue
            if message.get("error"):
                self.live_audio_error = message["error"]
                self.stream_audio_hint.setText(self.live_audio_error)
            elif message.get("running"):
                self.stream_audio_hint.setText(
                    f"Streaming audio · {message['delay_ms']} ms delay. Choose the cable's recording endpoint in your streaming app."
                )

    def _stream_audio_process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.live_audio_error = "Could not start the streaming audio worker."
            self._stream_audio_finished()

    def _stream_audio_finished(self, *_):
        process = self.live_audio
        if process:
            self._stream_audio_messages()
            if process.exitCode() and not self.live_audio_error and not self.live_audio_stopping:
                self.live_audio_error = "Streaming audio stopped unexpectedly: " + bytes(process.readAllStandardError()).decode(errors="replace")[-800:]
            process.deleteLater()
        self.live_audio = None
        self.audio_output.setEnabled(True)
        self._set_record_audio_enabled(not self.recorder)
        self.stream_audio_button.setText("Start streaming audio")
        self.stream_audio_hint.setText(self.live_audio_error or "Streaming audio stopped.")

    def _set_record_audio_enabled(self, enabled):
        enabled = enabled and not self.live_audio
        self.microphone.setEnabled(enabled)
        self.audio_delay.setEnabled(enabled)
        self.audio_delay_slider.setEnabled(enabled)

    def toggle_record(self):
        if self.record_saving:
            return
        if self.recorder:
            self.stop_record()
            return
        if not self.session:
            return
        mic = self.microphone.currentData()
        if mic is None:
            # The chosen microphone may have appeared after the startup scan
            # (for example, when Voicemod starts later). Refresh once on click.
            self.record_button.setEnabled(False)
            self.record_button.setText("Checking microphone…")
            self._task(
                lambda _: audio_devices(),
                self._record_microphones_ready,
                failed=self._record_start_failed,
            )
            return
        try:
            config = self.session.config
            self.recorder = Recorder(
                self.session.preview,
                config.width,
                config.height,
                self._destination(".mkv"),
                mic,
                audio_delay_ms=self.audio_delay.value(),
            )
            self.recorder.start()
        except Exception as exc:
            self.recorder = None
            self._record_start_failed(str(exc))
            return
        self.record_button.setText("■  Stop recording")
        self._set_record_audio_enabled(False)
        self.notice.setText(
            "Recording processed video" + (
                f" + {mic} · audio delay {self.recorder.audio_delay_ms} ms"
                if mic else " without audio"
            )
        )

    def _record_microphones_ready(self, microphones):
        self._audio_ready(microphones)
        self.record_button.setEnabled(self.session is not None)
        self.record_button.setText("●  Record")
        if not self.session:
            return
        if self.microphone.currentData() is None:
            self._record_start_failed(
                "No recording microphone is selected or the saved microphone is unavailable. "
                "Select a microphone in the left Audio panel, or choose Video only (no audio), then press Record again."
            )
            return
        self.toggle_record()

    def _record_start_failed(self, message):
        self.record_button.setEnabled(self.session is not None)
        self.record_button.setText("●  Record")
        self.notice.setText("Could not start recording: " + message)
        QMessageBox.warning(self, "Could not start recording", message)

    def stop_record(self):
        if not self.recorder or self.record_saving:
            return
        self.record_saving = True
        self.record_button.setText("Saving…")
        recorder = self.recorder

        def finish(signals):
            path = recorder.stop()
            if recorder.error:
                raise RuntimeError(recorder.error)
            return path

        self._task(finish, self._record_finished, failed=self._record_failed)

    def _record_failed(self, message):
        self.recorder = None
        self.record_saving = False
        self.record_button.setText("●  Record")
        self._set_record_audio_enabled(True)
        self.notice.setText("Recording failed: " + message)
        if self.pending_stop:
            self.pending_stop = False
            self.stop_camera()
        if self.pending_close:
            self.close()

    def _record_finished(self, path):
        self.recorder = None
        self.record_saving = False
        self.record_button.setText("●  Record")
        self._set_record_audio_enabled(True)
        self._export_ready(path)
        if self.pending_stop:
            self.pending_stop = False
            self.stop_camera()
        if self.pending_close:
            self.close()

    def _export_ready(self, path):
        self.last_export = Path(path)
        self.media_path = str(path)
        self.show_output.setEnabled(True)
        self.notice.setText(f"Saved: {path}")

    def choose_media(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open media for postwork",
            str(self.output_folder),
            (
                "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
                if self.mode == "Image"
                else "Videos (*.mkv *.mp4 *.mov *.avi)"
                if self.mode == "Video"
                else "Media (*.png *.jpg *.jpeg *.webp *.bmp *.mkv *.mp4 *.mov *.avi)"
            ),
        )
        if path:
            self.media_path = path
            self.notice.setText(f"Selected: {Path(path).name}")
            if not self.session:
                image = (
                    imread_unicode(path)
                    if Path(path).suffix.lower()
                    in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
                    else None
                )
                if image is not None:
                    self.preview.set_frame(image)
                else:
                    self.preview.set_frame(None)
                    self.preview.message = "Ready for the finishing touches"
                    self.preview.detail = Path(path).name
                    self.preview.update()

    def export_job(self):
        if self.session or self.recorder:
            self.notice.setText(
                "Stop the camera and recording before AI postwork to free GPU resources."
            )
            return
        if self.job_busy:
            return
        if not self.media_path:
            self.choose_media()
            if not self.media_path:
                return
        if not self.apply_enhance.isChecked() and not self.apply_swap.isChecked():
            self.notice.setText("Choose AI restoration or face replacement.")
            return
        if self.apply_swap.isChecked() and not self.source:
            self.notice.setText("Choose a source face first.")
            return
        image = Path(self.media_path).suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
        )
        default = self._destination(".png" if image else ".mp4", "FaceArt-finished")
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export a finished copy",
            str(default),
            "PNG (*.png)" if image else "MP4 (*.mp4)",
        )
        if not destination:
            return
        self.job_busy = True
        self.job_cancel = threading.Event()
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        model = self.model.currentText() if self.apply_enhance.isChecked() else None
        strength = self.strength.value() / 100
        source = self.media_path
        face = self.source if self.apply_swap.isChecked() else None
        self._task(
            lambda s: export_media(
                source,
                destination,
                model,
                strength,
                s.progress.emit,
                self.job_cancel,
                face,
            ),
            self._job_finished,
            self._job_progress,
        )

    def _job_progress(self, value, text):
        self.progress.setValue(value)
        self.notice.setText(text)

    def _job_finished(self, path):
        self.job_busy = False
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._export_ready(path)
        if self.pending_close:
            self.close()

    def cancel_export(self):
        self.job_cancel.set()
        self.notice.setText("Cancelling after the current operation…")

    def open_last(self):
        if self.last_export:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_export)))

    def open_folder(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_folder)))

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Recordings and exports", str(self.output_folder)
        )
        if path:
            self.output_folder = Path(path)
            self._save()
            self.notice.setText(f"Output folder: {path}")

    def open_projection(self):
        if not self.projection:
            self.projection = Preview()
            self.projection.setWindowTitle("FaceArt Studio · Processed projection")
            self.projection.resize(960, 540)
        self.projection.show()
        self.projection.raise_()

    def show_expression_plan(self):
        self.notice.setText("Expression controls are planned. The current webcam uses full face replacement without original-eye or mouth overlays.")

    def open_face_lab(self):
        if self.session or self.recorder or self.job_busy or self.live_load_busy:
            self.notice.setText("Stop the camera and finish current jobs before opening Face Lab.")
            return
        from modules.studio.face_lab import FaceLab
        import gc

        self.live_restorer = None
        gc.collect()
        dialog = FaceLab(self)
        dialog.exec()
        dialog.deleteLater()

    def _tick(self):
        if self.background_config.mode=='slideshow':
            self.slideshow.tick()
            if getattr(self,'_last_slide_active',None)!=self.slideshow.active:
                self._last_slide_active=self.slideshow.active
                self._save()
        now = time.perf_counter()
        if self.session:
            sequence, frame = self.session.preview()
            if (
                frame is not None
                and sequence != self.sequence
                and now - self.last_preview >= 1 / 15
            ):
                if self.preview_enabled.isChecked() and not self.isMinimized():
                    self.preview.set_frame(frame)
                if self.projection and self.projection.isVisible():
                    self.projection.set_frame(frame)
                self.sequence = sequence
                self.last_preview = now
            if now - self.last_metrics >= 1:
                m = self.session.metrics()
                output=m.get('output_device') or {}
                if m['output_error']:
                    self._last_output_error=m['output_error']
                    self.virtual.blockSignals(True);self.virtual.setChecked(False);self.virtual.blockSignals(False)
                    self.virtual.setText('Start virtual camera')
                    self.virtual_status.setText('Output off: '+m['output_error'])
                elif output.get('running'):
                    if self.notice.text()==getattr(self,'_last_output_error',None):
                        self.notice.setText('Local virtual camera is sending.')
                    self.virtual_status.setText(f"Sending locally · {output['device']} · {output['width']}×{output['height']} · nominal 30 FPS. Fresh processing: {m['recent_safe_fps']:.1f} FPS")
                elif not m['output_enabled']:
                    self.virtual_status.setText('Off · select OBS Virtual Camera in the receiving app after starting.')
                self.last_metrics = now
                self.camera_status.setText(
                    "●  " + ("Output error" if m["output_error"] else m["status"])
                )
                self.fps_label.setText(f"{m['recent_safe_fps']:.1f} FPS")
                self.timing.setText(
                    f"Processing {m['processing_p50_ms'] or '—'} ms  ·  Frame age {m['frame_age_p95_ms'] or '—'} ms"
                )
                if self.live_gpen.isChecked() and self.live_restorer is not None:
                    error = self.session.processor.enhancement_error
                    self.live_gpen_status.setText(
                        ("Using swap without enhancement: " + error)
                        if error
                        else f"GPEN 256 · {self.live_strength.value()}% · {self.session.processor.timings.get('enhance_ms', 0):.0f} ms"
                    )
                self.spark.values.append(m["recent_safe_fps"])
                self.spark.update()
                if m["output_error"]:
                    self.notice.setText(m["output_error"])
                elif self.notice.text() == "Loading face models." and m.get("safe_frames", 0):
                    self.notice.setText("Live camera ready.")
                elif self.notice.text() == "Loading face models." and m["status"].startswith("Model setup failed:"):
                    self.notice.setText(m["status"])
                elif (
                    m.get("source_status")
                    and m["source_status"] != self.last_source_status
                ):
                    self.last_source_status = m["source_status"]
                    self.notice.setText(m["source_status"])
        if self.session and self.background_config.mode != "off":
            error = self.session.processor.background_error
            self.background_status.setText(
                "Background stopped: " + error if error else
                (self.slideshow.status() if self.background_config.mode=='slideshow' else
                 f"Background active · {self.session.processor.timings.get('background_ms', 0):.0f} ms")
            )
        if self.recorder and not self.record_saving:
            if self.recorder.error:
                self.notice.setText("Recording failed: " + self.recorder.error)
                self.recorder = None
                self.record_button.setText("●  Record")
                self._set_record_audio_enabled(True)
            elif self.recorder.started:
                elapsed = int(now - self.recorder.started)
                self.record_button.setText(f"■  {elapsed // 60:02}:{elapsed % 60:02}")
        self.tasks = [(t, s) for t, s in self.tasks if t.is_alive()]

    def _shutdown(self):
        self.slideshow.close()
        self.live_audio_stopping = True
        if self.live_audio:
            from modules.studio.process_utils import stop_worker
            stop_worker(self.live_audio)
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        if self.session:
            self.session.stop()
            self.session = None
        if self.projection:
            self.projection.close()

    def closeEvent(self, event):
        if self.job_busy:
            self.pending_close = True
            self.cancel_export()
            event.ignore()
            return
        if self.recorder:
            self.pending_close = True
            self.stop_record()
            event.ignore()
            return
        self._shutdown()
        self._save()
        event.accept()
        QApplication.instance().quit()


def main(source=None):
    from modules.app_instance import acquire_gui_instance

    if not acquire_gui_instance("FaceArt Studio v0.2 experimental"):
        return
    from modules.studio.taskbar import configure_process, configure_window

    configure_process()
    app = QApplication.instance() or QApplication(sys.argv)
    from isolated_runtime import configure_fonts
    configure_fonts(app)
    app.setApplicationName("FaceArt Studio")
    app.setStyleSheet(STYLE)
    icon = Path(__file__).with_name("assets") / "faceart.ico"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    window = Studio(source)
    configure_window(window)
    window.show()
    app.exec()
