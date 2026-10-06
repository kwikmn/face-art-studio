"""Load the project environment's GPU libraries without importing the legacy UI."""
import os
from pathlib import Path
import sys

_DLL_HANDLES = []


def prepare_runtime():
    root = Path(__file__).resolve().parent
    os.environ['PATH'] = str(root / 'tools' / 'ffmpeg' / 'bin') + os.pathsep + os.environ.get('PATH', '')
    if sys.platform != 'win32':
        return
    site = Path(sys.prefix) / 'Lib' / 'site-packages'
    for directory in [site / 'torch' / 'lib', *sorted((site / 'nvidia').glob('*/bin'))]:
        if directory.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    import onnxruntime as ort
    if hasattr(ort, 'preload_dlls'):
        # Prefer the pinned NVIDIA wheels; stale optional torch metadata must not
        # redirect the loader to an absent torch/lib directory.
        ort.preload_dlls(directory='')
