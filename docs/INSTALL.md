# Windows installation and startup

## Supported configuration

This release targets **Windows x64, Python 3.10.6+ in the 3.10 series, and an NVIDIA CUDA-capable GPU with a working NVIDIA driver**. Candidate 5's live processor explicitly requests CUDA and rejects fallback; there is no CPU/AMD live selector in this release. The source's earlier macOS/Linux code is retained, but this tester setup and OBS output path support Windows only.

The original environment used Python 3.10.6, NumPy 1.26.4, ONNX Runtime GPU 1.21.0, OpenCV 4.10, InsightFace 0.7.3, PySide6 6.11.2 and pyvirtualcam 0.14.0. `requirements.txt`, `requirements-cuda.txt` and `constraints.txt` pin the setup to that runtime family. CUDA 12.8/cuDNN 9.10 libraries are installed into the local Python environment from NVIDIA's official PyPI packages; copying a CUDA toolkit into the source is unnecessary.

Allow at least **8 GiB free** for the base environment, unpacking and required models, plus room for recordings. The optional Face Lab requires substantially more storage and VRAM. Use a short writable path outside Program Files; spaces in folder names are supported. Keep the whole extracted folder together, and do not run directly inside the ZIP archive. A Python venv is not portable between machines; recreate it by running setup on the destination machine.

## Prerequisites — install manually once

1. **Python 3.10 x64**, including pip and the Python launcher. [Python 3.10.11's official Windows installer page](https://www.python.org/downloads/release/python-31011/) provides a conventional installer; later 3.10 security releases may be source-only. Other Python series are rejected because this build's dependencies were derived from 3.10. The Python launcher avoids needing Python on PATH. Use current vendor security guidance when deciding which Python distribution to install.
2. **NVIDIA GPU driver** from [NVIDIA's driver page](https://www.nvidia.com/Download/index.aspx). Setup reads `nvidia-smi` and the runtime check executes a small synthetic ONNX model through CUDA. A listed provider alone is not treated as proof of working CUDA. The original workstation used driver 581.57; other configurations remain tester-dependent. See [ONNX Runtime's CUDA compatibility documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
3. **Microsoft Visual C++ x64 Redistributable** from [Microsoft's supported download page](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist). It supplies native runtime DLLs used by ONNX Runtime and Qt.
4. **Visual Studio 2022 Build Tools** from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Select **Desktop development with C++**, the **MSVC v143 x64/x86 toolset** and a **Windows 10/11 SDK**. [InsightFace 0.7.3 on PyPI](https://pypi.org/project/insightface/0.7.3/) ships source, so this setup builds its extension locally instead of using an unofficial wheel. The setup detects the C++ toolset and stops before package installation if absent.
5. **For virtual webcam output:** install [OBS Studio for Windows](https://obsproject.com/download) using its installer so **OBS Virtual Camera** is registered. Candidate 5 was tested with the installed OBS 32.0.2 component and pyvirtualcam 0.14.0. Installing OBS can require administrator approval; do that yourself. FaceArt never registers or installs the driver. **The OBS application does not need to run**, and OBS's own virtual-camera producer must be stopped. [pyvirtualcam's documented Windows setup](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras) explains this driver dependency.
6. **For recording, audio devices and postwork export:** obtain a Windows FFmpeg build linked from the [official FFmpeg download page](https://ffmpeg.org/download.html). Put `ffmpeg.exe` and `ffprobe.exe` in `tools/ffmpeg/bin/`, or add their folder to your PATH yourself. These binaries are not bundled. The runtime adds the project tools directory for its process only. Run `ffmpeg -encoders` and confirm `h264_nvenc` for the current live recorder. FFmpeg build licenses vary; keep the chosen build's license notices.

The Python setup runs as a normal user. Some vendor prerequisite installers may need elevation; FaceArt neither invokes them nor requests elevation. No PowerShell execution-policy changes are necessary: the launchers use Python directly.

## Setup and start

Double-click `Setup-FaceArt.cmd`. It looks for Python 3.10 through the launcher, PATH, then standard per-user/Program Files installation locations. The script checks prerequisites and free space, creates `venv/`, installs pinned packages from **https://pypi.org/simple**, runs `pip check`, imports the actual Studio and performs synthetic CUDA inference using the bundled non-sensitive segmentation model. It does not construct the Studio window or open any device. Package logs are in `state/logs/setup-<UTC>.log`.

The installer uses argument lists for paths with spaces, ignores inherited pip indexes/configuration, and installs into the project environment. It uses the existing `PIP_CACHE_DIR` if you configured one; otherwise setup downloads are cached in `state/cache/pip`. For a roomy secondary drive you can set the cache for the current terminal only before launching:

```powershell
$env:PIP_CACHE_DIR = 'F:\Caches\pip'
.\Setup-FaceArt.cmd
```

If models are missing, a **manual model action** message and exit code 2 follow successful Python package installation. This is expected on first setup. Follow [MODELS.md](MODELS.md), run `Check-FaceArt.cmd` again, then double-click `Start-FaceArt.cmd`. Optional model files are reported only if damaged; they are not required for baseline live swapping. Setup does not download or accept terms for restricted weights.

Rerunning setup preserves `models/`, `state/` and a compatible `venv/`. It rechecks package requirements; it does not delete or reset existing settings. If a venv is incomplete, uses another Python version, or was copied from another machine, rename that **venv directory only** yourself and rerun setup. Preserve models and state. Avoid running setup while Studio is open.

Useful terminal commands:

```powershell
.\Setup-FaceArt.cmd --check-only  # prerequisites only, no installs
.\Setup-FaceArt.cmd --dry-run     # prerequisites and exact planned commands
.\Setup-FaceArt.cmd --verify-only # existing environment plus model checks
.\venv\Scripts\python.exe -s scripts\check_install.py --hash-models
.\Start-FaceArt.cmd --source "C:\My Faces\replacement.png"
```

Create an ordinary Windows shortcut to `Start-FaceArt.cmd` if desired. Set its Start in folder to the release folder. No shortcut is created automatically.

## First preview and receiving apps

1. Start Studio. Choose a clear face image you have permission to use and a physical camera input.
2. Choose 540p first, then **Start camera**. Inspect the processed preview and its fresh processing performance before increasing resolution or enabling enhancement.
3. Select a background mode/backend. RVM/MODNet need their separate files; MediaPipe is bundled. Chair suppression works on MediaPipe with replacement/blur only, and may remove hair or arms at strong settings.
4. Explicitly click **Start virtual camera**. Output is off on startup, after camera restart, and after stopping the camera. Stop it when done.
5. In browser/Discord video settings, choose **OBS Virtual Camera**. Allow that application's camera access manually if needed. Use a local preview for first acceptance. Browser/Discord menus vary, and actual receipt has not been checked for this exact release.
6. Select **Voicemod** separately in the receiving app's microphone menu. FaceArt's virtual output carries no audio. Optional FaceArt streaming audio/delay requires an independently installed audio cable/endpoint; it is not needed for selecting Voicemod normally.

Do not start another OBS virtual-camera producer concurrently. A busy-device error means you should stop the producer yourself; FaceArt does not stop OBS or reconfigure another app. If the error persists, close/reopen stale receivers and retry.

## Troubleshooting

| Message/symptom | Action |
| --- | --- |
| Python missing or wrong version/architecture | Install Python 3.10 x64 with its launcher; run `py -3.10 --version`. |
| MSVC compiler/C++ Build Tools error | Add MSVC v143 and the Windows SDK in the Build Tools installer, then rerun setup. Do not substitute an unverified InsightFace wheel. |
| Package/network install error | Read `state/logs/setup-*.log`; check connectivity to PyPI, available space and prerequisites, then rerun. Setup is repeatable. |
| `CUDA unavailable`, DLL error, runtime falls back | Update the NVIDIA driver, install VC++ x64 runtime, rerun setup and `Check-FaceArt.cmd`. Installing `onnxruntime` alongside `onnxruntime-gpu` can overwrite the module; use the isolated setup environment. |
| Missing model/wrong file size/hash | Follow `MODELS.md`; ZIP contents must not add an extra nesting level. HTML pages and Git LFS pointers are not model weights. |
| RVM/MODNet unavailable | Select MediaPipe until its optional download is placed correctly. CUDA matting requires successful CUDA runtime checks. |
| OBS device absent in browser/Discord | Install/register OBS's Windows virtual-camera component with the official OBS installer and restart the receiving app. Python packages cannot supply this driver. |
| Virtual device busy | Stop the other virtual-camera producer and retry; close/reopen a stale receiver if needed. |
| FFmpeg missing, recording/export failure | Place both executables in `tools/ffmpeg/bin/`, verify the chosen build includes NVENC, and check driver support. |
| Preview slow or jittering | Use 540p, disable GPEN, select MediaPipe or lower RVM ratio, and inspect **fresh** processing FPS. A 30 Hz output setting can repeat frames. |
| Startup exits | `Start-FaceArt.cmd` prints the local `state/logs/studio-*.log` location. Run `Check-FaceArt.cmd`. Review logs for private paths/device names before sending them. |

## Optional Face Lab

Face Lab code is retained, but its local generation runtime is **not installed by baseline setup**. It needs a suitable NVIDIA GPU supporting BF16 and substantially more VRAM/storage. The development environment used **PyTorch 2.9.1 CUDA 12.8**, plus the pinned packages in `requirements-facegen.txt`. Install optional packages only if you want generation:

```powershell
.\venv\Scripts\python.exe -m pip --isolated install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
.\venv\Scripts\python.exe -m pip --isolated install --index-url https://pypi.org/simple -r requirements-facegen.txt
.\venv\Scripts\python.exe -m pip check
```

The [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/) identify the CUDA index. Optional dependencies can change shared dependency versions; rerun baseline setup/checks if needed. Follow the [FLUX model location and revision](MODELS.md#optional-face-lab-model). Downloads/terms are separate, and this optional clean installation has not been exercised here. If model access needs an account or permission, obtain it yourself on the provider site; setup handles no credentials.

The retained `run.py` legacy Deep-Live-Cam interface is not the tester entry point and can require additional legacy packages such as TensorFlow/opennsfw2. Use `Start-FaceArt.cmd` for the documented build.
