"""App-focused playlist controls; removing entries never touches source files."""

from pathlib import Path
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QDoubleSpinBox,
    QCheckBox,
)


class SlideshowDialog(QDialog):
    changed = Signal()
    activate = Signal()

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Live background slideshow — v0.2 experiment")
        self.resize(740, 600)
        self.setStyleSheet(
            "QDialog { background:#151d21; } QListWidget { background:#101619; color:#e7eeed; border:1px solid #344149; padding:6px; } QDoubleSpinBox { background:#192226; color:#e7eeed; border:1px solid #344149; padding:5px; }"
        )
        layout = QVBoxLayout(self)
        title = QLabel("Background slideshow")
        title.setStyleSheet("font-size:24px;font-weight:600;")
        layout.addWidget(title)
        hint = QLabel(
            "Drag images to reorder. Source files stay in place. Background fades only; foreground stays live."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.playlist = QListWidget()
        self.playlist.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.playlist.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        for path in controller.paths:
            self.add_row(path)
        layout.addWidget(self.playlist)
        row = QHBoxLayout()
        layout.addLayout(row)
        for text, action in [
            ("Add images…", self.add_images),
            ("Remove selected", self.remove_selected),
            ("Use selected", self.use_selected),
        ]:
            button = QPushButton(text)
            button.clicked.connect(action)
            row.addWidget(button)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.previous = QPushButton("Previous")
        self.previous.clicked.connect(lambda: self.navigate(-1))
        row.addWidget(self.previous)
        self.play = QPushButton("Play / resume")
        self.play.clicked.connect(self.toggle)
        row.addWidget(self.play)
        self.next = QPushButton("Next")
        self.next.clicked.connect(lambda: self.navigate(1))
        row.addWidget(self.next)
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(controller.loop)
        row.addWidget(self.loop)
        row = QHBoxLayout()
        layout.addLayout(row)
        row.addWidget(QLabel("Interval (seconds)"))
        self.interval = QDoubleSpinBox()
        self.interval.setRange(1, 3600)
        self.interval.setDecimals(1)
        self.interval.setValue(controller.interval)
        row.addWidget(self.interval)
        row.addWidget(QLabel("Background fade (0 = instant)"))
        self.fade = QDoubleSpinBox()
        self.fade.setRange(0, 2)
        self.fade.setSingleStep(0.1)
        self.fade.setDecimals(2)
        self.fade.setValue(controller.fade)
        row.addWidget(self.fade)
        keys = QLabel(
            "While Studio has focus: Ctrl+Alt+Left / Right = previous / next; Ctrl+Alt+Space = play / pause."
        )
        keys.setWordWrap(True)
        layout.addWidget(keys)
        note = QLabel(
            "Images: max 32 MB, 24 megapixels, 20000 px edge; up to 200 entries. Missing/bad slides keep last valid background. Restart opens paused."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.playlist.model().rowsMoved.connect(self.order_changed)
        self.playlist.itemDoubleClicked.connect(lambda *_: self.use_selected())
        self.interval.valueChanged.connect(self.configure)
        self.fade.valueChanged.connect(self.configure)
        self.loop.toggled.connect(self.configure)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        self.refresh()
        for key, callback in [
            ("Ctrl+Alt+Left", lambda: self.navigate(-1)),
            ("Ctrl+Alt+Right", lambda: self.navigate(1)),
            ("Ctrl+Alt+Space", self.toggle),
        ]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)

    def add_row(self, path):
        item = QListWidgetItem(Path(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self.playlist.addItem(item)

    def order_changed(self, *_):
        self.controller.set_paths(
            [
                self.playlist.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.playlist.count())
            ]
        )
        self.changed.emit()

    def add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add slideshow images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        existing = set(self.controller.paths)
        for path in paths:
            if path not in existing and self.playlist.count() < 200:
                self.add_row(path)
                existing.add(path)
        self.order_changed()

    def remove_selected(self):
        for item in self.playlist.selectedItems():
            self.playlist.takeItem(self.playlist.row(item))
        self.order_changed()

    def use_selected(self):
        item = self.playlist.currentItem()
        if item:
            self.controller.select(item.data(Qt.ItemDataRole.UserRole))
            self.activate.emit()
            self.changed.emit()

    def navigate(self, step):
        self.controller.navigate(step)
        self.activate.emit()
        self.changed.emit()

    def toggle(self):
        self.controller.play(not self.controller.playing)
        self.activate.emit()
        self.changed.emit()
        self.refresh()

    def configure(self, *_):
        self.controller.configure(
            self.interval.value(), self.fade.value(), self.loop.isChecked()
        )
        self.changed.emit()

    def refresh(self):
        self.status.setText(self.controller.status())
        self.play.setText("Pause" if self.controller.playing else "Play / resume")
        self.previous.setEnabled(bool(self.controller.paths))
        self.next.setEnabled(bool(self.controller.paths))
        self.play.setEnabled(bool(self.controller.paths))
