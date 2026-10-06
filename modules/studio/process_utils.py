"""Stop only a QProcess and its owned Windows launcher children."""

import psutil
from PySide6.QtCore import QProcess


def stop_worker(process):
    if process.state() == QProcess.ProcessState.NotRunning:
        return
    pid = int(process.processId())
    if pid > 0:
        try:
            children = psutil.Process(pid).children(recursive=True)
            for child in reversed(children):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    process.kill()
    process.waitForFinished(3000)
