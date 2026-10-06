"""Run independent fixtures without legacy import-stub leakage between files."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
environment = os.environ.copy()
environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', NO_ALBUMENTATIONS_UPDATE='1')
failed = []
for path in sorted((ROOT / 'tests').glob('test_*.py')):
    print(f'\n{path.name}', flush=True)
    result = subprocess.call([sys.executable, '-s', '-m', 'unittest', 'discover', '-s', 'tests', '-p', path.name], cwd=ROOT, env=environment)
    if result:
        failed.append(path.name)
if failed:
    print('Failed:', ', '.join(failed))
    raise SystemExit(1)
print('\nAll requested fixtures passed (optional skips are reported above).')
