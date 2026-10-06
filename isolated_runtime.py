"""All experimental mutable state stays inside this project."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'state'
for name in ('home', 'local', 'roaming', 'temp', 'cache', 'outputs'):
    (STATE / name).mkdir(parents=True, exist_ok=True)
for key, value in {
    'HOME': STATE / 'home', 'USERPROFILE': STATE / 'home',
    'LOCALAPPDATA': STATE / 'local', 'APPDATA': STATE / 'roaming',
    'TMP': STATE / 'temp', 'TEMP': STATE / 'temp',
    'HF_HOME': STATE / 'cache' / 'huggingface',
    'TORCH_HOME': STATE / 'cache' / 'torch',
    'XDG_CACHE_HOME': STATE / 'cache', 'PIP_CACHE_DIR': STATE / 'cache' / 'pip',
    'MPLCONFIGDIR': STATE / 'cache' / 'matplotlib',
}.items():
    os.environ[key] = str(value)
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
if os.name == 'nt':
    _dll_handles = []
    for directory in (ROOT / 'runtime' / 'cuda',):
        if directory.exists():
            _dll_handles.append(os.add_dll_directory(str(directory)))
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ['PATH']

def configure_fonts(app):
    from PySide6.QtGui import QFontDatabase, QFont
    font = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'segoeui.ttf'
    if font.is_file():
        QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont('Segoe UI', 10))
