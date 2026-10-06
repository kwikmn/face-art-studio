"""Check packages, synthetic ONNX execution and model placement; no device capture."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def model_problems(hash_models=False, required_only=False):
    manifest = json.loads((ROOT / 'models/downloads.json').read_text(encoding='utf-8'))
    problems = []
    for item in manifest['models']:
        if required_only and not item.get('required'):
            continue
        path = ROOT / item['path']
        if not path.is_file():
            if item.get('required'):
                problems.append(f"Missing {item['path']}\n  Download: {item['url']}")
            continue
        if item.get('bytes') and path.stat().st_size != item['bytes']:
            problems.append(f"Wrong size: {item['path']}. Download the ONNX file, not an HTML page or Git LFS pointer.")
        elif hash_models and item.get('sha256'):
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                problems.append(f"Checksum mismatch: {item['path']}; see docs/MODELS.md.")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages-only', action='store_true')
    parser.add_argument('--models-only', action='store_true')
    parser.add_argument('--hash-models', action='store_true')
    args = parser.parse_args()
    problems = []
    if not args.models_only:
        try:
            import isolated_runtime  # noqa: F401
            from runtime_bootstrap import prepare_runtime
            prepare_runtime()
            for name in ['numpy', 'cv2', 'onnx', 'insightface', 'PySide6', 'PIL', 'psutil', 'requests', 'pygrabber', 'pyvirtualcam', 'cv2_enumerate_cameras']:
                importlib.import_module(name)
            # Import the actual Studio without constructing its window or enumerating devices.
            importlib.import_module('modules.studio.app')
            import numpy as np
            import onnxruntime as ort
            print('Python:', sys.version.split()[0])
            for name in ['numpy', 'onnxruntime-gpu', 'insightface', 'PySide6', 'pyvirtualcam']:
                print(name + ':', importlib.metadata.version(name))
            providers = ort.get_available_providers()
            print('Available ONNX providers:', ', '.join(providers))
            if 'CUDAExecutionProvider' not in providers:
                raise RuntimeError('CUDA provider unavailable. Rerun setup on a supported NVIDIA system.')
            model = ROOT / 'modules/studio/assets/background/mediapipe.onnx'
            session = ort.InferenceSession(str(model), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            if 'CUDAExecutionProvider' not in session.get_providers():
                raise RuntimeError('CUDA libraries/driver failed to load. Install/update the NVIDIA driver and Microsoft VC++ x64 Redistributable, then rerun setup.')
            result = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 144, 256, 3), dtype=np.float32)})[0]
            if result.shape != (1, 144, 256, 2) or not np.isfinite(result).all():
                raise RuntimeError('Synthetic ONNX check returned invalid output.')
            print('Synthetic CUDA execution passed. No camera, microphone or virtual output was opened.')
        except Exception as error:
            problems.append(f'Runtime check failed: {error}')
    if not args.packages_only:
        problems.extend(model_problems(args.hash_models))
    for problem in problems:
        print('ACTION NEEDED:', problem)
    if problems:
        print('See docs/INSTALL.md and docs/MODELS.md. Run Check-FaceArt.cmd again after fixing these items.')
        return 2
    print('Requested checks passed.')
    if not args.models_only:
        print('Virtual output also needs the registered OBS Virtual Camera driver. Install OBS manually; the OBS app does not need to run. See docs/INSTALL.md.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
