"""Repeatable, project-local setup. No drivers, credentials or system changes."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / 'venv'
PYTHON = VENV / 'Scripts' / 'python.exe'
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
# Leave room below Windows' traditional 260-character compiler path limit
# for pip's unpack directory plus InsightFace's nested object-file paths.
MAX_BUILD_TEMP_PATH = 64


def supported_python():
    return sys.version_info[:2] == (3, 10) and sys.version_info[:3] >= (3, 10, 6) and struct.calcsize('P') == 8


def supported_windows(version=None):
    if sys.platform != 'win32':
        return False
    version = version or sys.getwindowsversion()
    return version.major == 10 and (version.product_type == 1 or version.build == 20348)


def supported_driver(gpu):
    """Use the Windows minimum for the pinned CUDA 12.8/cuDNN 9.10.2 stack."""
    try:
        versions = [tuple(int(part) for part in line.rsplit(',', 1)[1].strip().split('.')) for line in gpu.splitlines()]
    except (IndexError, ValueError):
        return False
    return bool(versions) and all(len(version) == 2 and version >= (570, 65) for version in versions)


def build_temp_base():
    base = Path(tempfile.gettempdir()).resolve()
    if sys.platform == 'win32' and len(str(base)) + len('/fas-12345678') > MAX_BUILD_TEMP_PATH:
        buffer = ctypes.create_unicode_buffer(32768)
        size = ctypes.windll.kernel32.GetShortPathNameW(str(base), buffer, len(buffer))
        if 0 < size < len(buffer):
            base = Path(buffer.value)
    if len(str(base)) + len('/fas-12345678') > MAX_BUILD_TEMP_PATH:
        raise RuntimeError('The build temp path is too long for the InsightFace compiler. Set TMP and TEMP to an existing short writable folder (base path at most 51 characters), then rerun setup. Do not change Windows long-path settings.')
    return base


def command(arguments, log):
    # Argument lists handle spaces without shell interpolation.
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='', NO_ALBUMENTATIONS_UPDATE='1')
    # --isolated alone still reads global/site pip configuration files.
    environment['PIP_CONFIG_FILE'] = os.devnull
    # Honor a configured pip cache (for example a roomy secondary disk).
    environment.setdefault('PIP_CACHE_DIR', str(ROOT / 'state' / 'cache' / 'pip'))
    # Build outside the possibly deep project path. Cleanup covers only this
    # uniquely owned directory; logs and project cache/state stay in place.
    with tempfile.TemporaryDirectory(prefix='fas-', dir=build_temp_base(), ignore_cleanup_errors=True) as build_temp:
        environment['TMP'] = environment['TEMP'] = build_temp
        execute(arguments, log, environment)


def execute(arguments, log, environment):
    print('Running:', subprocess.list2cmdline([str(a) for a in arguments]), flush=True)
    # --isolated plus the null config excludes inherited indexes/config files.
    # Do not log the inherited environment or user configuration.
    with subprocess.Popen([str(a) for a in arguments], cwd=ROOT, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', creationflags=HIDDEN) as process:
        for line in process.stdout:
            print(line.rstrip(), flush=True)
            log.write(line)
            log.flush()
        result = process.wait()
    if result:
        raise RuntimeError(f'Command failed (exit {result}). See the setup log. For InsightFace compiler errors, check the quoted MSVC/SDK error and build-path guidance in docs/INSTALL.md. If Build Tools already passed preflight, reinstalling them is not the first remedy.')


def nvidia_gpu():
    executable = shutil.which('nvidia-smi')
    if not executable:
        executable = str(Path(os.environ.get('WINDIR', 'C:/Windows')) / 'System32' / 'nvidia-smi.exe')
    try:
        result = subprocess.run([executable, '--query-gpu=name,driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=15, creationflags=HIDDEN)
    except (OSError, subprocess.TimeoutExpired):
        return ''
    return result.stdout.strip() if result.returncode == 0 else ''


def cpp_tools():
    vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    if not vswhere.is_file():
        return False
    try:
        result = subprocess.run([str(vswhere), '-latest', '-products', '*', '-requires',
                                 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'],
                                capture_output=True, text=True, timeout=15, creationflags=HIDDEN)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip()) and result.returncode == 0


def existing_insightface():
    """A compatible installed extension can be reused; no compiler is needed."""
    if not PYTHON.is_file():
        return False
    probe = (
        "import importlib.metadata as m; "
        "d=m.distribution('insightface'); "
        "ok=d.version=='0.7.3' and any(str(p).endswith('.pyd') and "
        "'mesh_core_cython' in str(p) and d.locate_file(p).is_file() for p in d.files or []); "
        "raise SystemExit(0 if ok else 1)"
    )
    try:
        result = subprocess.run([str(PYTHON), '-s', '-c', probe], capture_output=True,
                                text=True, timeout=15, creationflags=HIDDEN)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def obs_registered():
    if sys.platform != 'win32':
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r'CLSID\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}\InprocServer32') as key:
            driver, _ = winreg.QueryValueEx(key, '')
            return Path(driver).is_file()
    except OSError:
        return False


def preflight():
    problems = []
    if not supported_windows():
        problems.append('The pinned CUDA/cuDNN runtime supports Windows 10, Windows 11 or Windows Server 2022 x64. See docs/INSTALL.md.')
    if not supported_python():
        problems.append('Use Python 3.10.6 or newer in the 3.10 series (64-bit). Install the Python launcher; see docs/INSTALL.md.')
    try:
        base = build_temp_base()
        print('Short build temp base:', base)
        if shutil.disk_usage(base).free < 3 * 1024**3:
            problems.append('Free at least 3 GiB on the build-temp drive for unpacking/compilation, or set TMP and TEMP to an existing short writable folder on a roomier drive.')
    except (OSError, RuntimeError) as error:
        problems.append(str(error))
    free = shutil.disk_usage(ROOT).free / (1024**3)
    print(f'Free space on the release drive: {free:.1f} GiB')
    if free < 12:
        problems.append('Free at least 12 GiB on the release drive for installation and required models; 15 GiB or more is recommended for OS headroom. Move the extracted source to a roomier drive if needed. Optional Face Lab and recordings need considerably more.')
    cache = Path(os.environ.get('PIP_CACHE_DIR', str(ROOT / 'state/cache/pip'))).expanduser().resolve()
    cache_ancestor = cache
    while not cache_ancestor.exists() and cache_ancestor.parent != cache_ancestor:
        cache_ancestor = cache_ancestor.parent
    try:
        if shutil.disk_usage(cache_ancestor).free < 3 * 1024**3:
            problems.append('Free at least 3 GiB on the pip-cache drive, or set PIP_CACHE_DIR to a folder on a roomier drive. See docs/INSTALL.md.')
    except OSError as error:
        problems.append(f'Cannot inspect the selected pip-cache drive: {error}')
    gpu = nvidia_gpu()
    print('NVIDIA GPU/driver:', gpu or 'not detected')
    if not gpu:
        problems.append('Candidate 5 live processing requires an NVIDIA GPU and driver. Install/update the driver manually from nvidia.com; CPU-only live mode is not exposed in this build.')
    elif not supported_driver(gpu):
        problems.append('Use NVIDIA Windows driver 570.65 or newer for the pinned CUDA 12.8/cuDNN 9.10.2 runtime. The detected version is older or could not be read; update/check nvidia-smi before installing packages. See docs/INSTALL.md.')
    reusable = existing_insightface()
    if reusable:
        print('Reusing installed InsightFace 0.7.3 extension; C++ Build Tools are not needed for this rerun.')
    elif not cpp_tools():
        problems.append('InsightFace 0.7.3 is built from the official PyPI source. Install Visual Studio 2022 Build Tools: Desktop development with C++, MSVC v143 and Windows SDK. See docs/INSTALL.md.')
    print('OBS Virtual Camera driver:', 'registered' if obs_registered() else 'missing; install OBS Studio manually for virtual output (docs/INSTALL.md)')
    if sys.platform == 'win32':
        for name in ['vcruntime140.dll', 'msvcp140.dll']:
            try:
                ctypes.WinDLL(name)
            except OSError:
                problems.append('Install the Microsoft Visual C++ x64 Redistributable from the official link in docs/INSTALL.md.')
                break
    for name in ['ffmpeg', 'ffprobe']:
        found = shutil.which(name) or (ROOT / 'tools' / 'ffmpeg' / 'bin' / f'{name}.exe').is_file()
        if not found:
            print(f'Optional feature prerequisite missing: {name}. Recording, audio and export require FFmpeg; see docs/INSTALL.md.')
    if VENV.exists() and not PYTHON.is_file():
        problems.append('The existing venv is incomplete. Rename it yourself, then rerun setup. Models and state will be preserved.')
    if PYTHON.is_file():
        try:
            result = subprocess.run([str(PYTHON), '-s', '-c', 'import sys,struct; print(sys.version_info[:2],struct.calcsize("P"))'], capture_output=True, text=True, timeout=15, creationflags=HIDDEN)
            compatible = result.returncode == 0 and result.stdout.strip() == '(3, 10) 8'
        except (OSError, subprocess.TimeoutExpired):
            compatible = False
        if not compatible:
            problems.append('The existing venv uses an incompatible interpreter. Rename it yourself, then rerun setup with Python 3.10 x64.')
    return problems


def install_plan():
    pip = [str(PYTHON), '-s', '-m', 'pip', '--isolated', '--disable-pip-version-check']
    index = ['--index-url', 'https://pypi.org/simple', '--cache-dir', os.environ.get('PIP_CACHE_DIR', str(ROOT / 'state/cache/pip'))]
    return [
        [sys.executable, '-s', '-m', 'venv', str(VENV)],
        [*pip, 'install', *index, 'pip==25.3', 'setuptools==75.8.0', 'wheel==0.45.1'],
        [*pip, 'install', *index, 'numpy==1.26.4', 'Cython==3.2.2'],
        [*pip, 'install', *index, '--no-build-isolation', '-c', str(ROOT / 'constraints.txt'), '-r', str(ROOT / 'requirements.txt'), '-r', str(ROOT / 'requirements-cuda.txt')],
        [*pip, 'check'],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Read-only prerequisite checks, no installation')
    parser.add_argument('--dry-run', action='store_true', help='Print prerequisites and planned commands, no installation')
    parser.add_argument('--verify-only', action='store_true', help='Check an existing environment, no installation')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.verify_only:
        if not PYTHON.is_file():
            print('No project environment. Run Setup-FaceArt.cmd first.')
            return 1
        return subprocess.call([str(PYTHON), '-s', str(ROOT / 'scripts/check_install.py')], cwd=ROOT)
    problems = preflight()
    for problem in problems:
        print('ACTION NEEDED:', problem)
    if args.dry_run:
        for arguments in install_plan():
            print('Plan:', subprocess.list2cmdline(arguments))
        return 1 if problems else 0
    if args.check_only or problems:
        return 1 if problems else 0
    logs = ROOT / 'state' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    path = logs / f'setup-{stamp}.log'
    print('Setup log:', path)
    try:
        with path.open('w', encoding='utf-8') as log:
            for index, arguments in enumerate(install_plan()):
                if index == 0 and PYTHON.is_file():
                    continue  # Never recreate a working environment or remove settings/models.
                command(arguments, log)
            command([str(PYTHON), '-s', str(ROOT / 'scripts/check_install.py'), '--packages-only'], log)
        (logs / 'installed.json').write_text(json.dumps({'runtime': 'v0.2-candidate5', 'python': '3.10', 'completed_utc': stamp}, indent=2) + '\n', encoding='utf-8')
        print('Python environment installed. Checking manually acquired models...')
        return subprocess.call([str(PYTHON), '-s', str(ROOT / 'scripts/check_install.py')], cwd=ROOT)
    except (OSError, RuntimeError) as error:
        print('Setup stopped:', error)
        print('Log:', path)
        print('Correct the prerequisite or connection problem, then rerun Setup-FaceArt.cmd. Existing state/models are preserved.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
