"""Preserve a local diagnostic log and refuse to start with missing live models."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_install import model_problems


def main():
    problems = model_problems(required_only=True)
    if problems:
        print('\n'.join(problems))
        print('Finish the manual model steps in docs/MODELS.md, then start again.')
        return 2
    logs = ROOT / 'state/logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    path = logs / f'studio-{stamp}.log'
    print('Studio log:', path, flush=True)
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='', NO_ALBUMENTATIONS_UPDATE='1')
    with path.open('w', encoding='utf-8') as log:
        result = subprocess.call([sys.executable, '-s', str(ROOT / 'run_studio.py'), *sys.argv[1:]], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result:
        print(f'Studio stopped (exit {result}). Read {path}; try Check-FaceArt.cmd and docs/INSTALL.md.')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
