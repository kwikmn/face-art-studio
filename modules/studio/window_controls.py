"""Native window movement from the Studio header."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFrame, QMenu


class WindowHeader(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("header")
        self.setToolTip("Drag to move the window. Double-click to maximize or restore.")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            if window.isMaximized():
                window.showNormal()
            else:
                window.showMaximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


def move_to_screen(window, screen):
    """Restore and center the window within the selected monitor's work area."""
    if screen not in QApplication.screens():
        return
    window.showNormal()
    area = screen.availableGeometry()
    handle = window.windowHandle()
    if handle is not None:
        handle.setScreen(screen)
    # Include native title bar and borders when fitting to the work area.
    frame = window.frameGeometry()
    border_w = max(0, frame.width() - window.width())
    border_h = max(0, frame.height() - window.height())
    window.resize(
        min(window.width(), area.width() - border_w),
        min(window.height(), area.height() - border_h),
    )
    frame = window.frameGeometry()
    frame.moveCenter(area.center())
    window.move(frame.topLeft())
    window.raise_()
    window.activateWindow()


def screen_menu(window, button):
    menu = QMenu(button)
    for index, screen in enumerate(QApplication.screens(), 1):
        current = screen == window.screen()
        text = f"Screen {index}: {screen.name()}" + (" (current)" if current else "")
        action = menu.addAction(text)
        action.triggered.connect(
            lambda checked=False, target=screen: move_to_screen(window, target)
        )
    menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
