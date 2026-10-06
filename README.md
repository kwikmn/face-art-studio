# FaceArt Studio v0.2 — tester build

Experimental Windows desktop face swapping, live enhancement, background replacement/slideshow and processed virtual camera output. This release is based on the working **v0.2 Candidate 5** source, with a separate setup workflow and no personal media or restricted model weights.

## Get running

1. Download **Code → Download ZIP** from this repository and extract it into a writable folder, or clone:
   ```powershell
   git clone https://github.com/kwikmn/face-art-studio.git
   cd face-art-studio
   ```
2. Follow [the Windows prerequisites](docs/INSTALL.md): Python 3.10 x64, an NVIDIA GPU/driver, Microsoft C++ Build Tools and the VC++ x64 runtime. Install OBS Studio once if you want the selectable virtual webcam. Add FFmpeg for recording, audio and export.
3. Double-click **Setup-FaceArt.cmd**. It creates a local `venv`, installs pinned packages from PyPI, and checks the runtime. It preserves existing models and settings. Setup may take several minutes and download over 1 GB of runtime libraries.
4. Obtain the required models using [the exact links and folder layout](docs/MODELS.md). Model terms apply separately; InsightFace pretrained weights are for non-commercial research. Setup checks these files and tells you what is missing.
5. Double-click **Check-FaceArt.cmd**, then **Start-FaceArt.cmd**. Choose your own replacement-face image and physical camera. Click **Start camera**, inspect the preview, then explicitly click **Start virtual camera** when ready.

This is **one-launch Python setup after prerequisites**, with guided model acquisition. It is not a self-contained installer: Python, Microsoft tools, the GPU driver and OBS Virtual Camera driver require separate installation. Setup never installs drivers, changes security policy, requests credentials or starts camera/microphone capture.

## Using the build

- Choose **OBS Virtual Camera** in the browser/Discord camera selector. FaceArt writes directly to the installed driver; **OBS does not need to run**. Do not run OBS's virtual camera producer at the same time. Browser camera permission may need enabling manually.
- Choose **Voicemod** separately as the receiving application's microphone. FaceArt virtual output is video only; Voicemod is not bundled.
- Background modes: off, blur, color, image and slideshow. Mask backends: bundled MediaPipe fallback, optional RVM and optional MODNet. Chair suppression/tightness refine the **MediaPipe fallback only**, while replacement or blur is active. Chair leakage and clipping of hair/arms can still occur.
- Live GPEN enhancement requires its optional model and CUDA. Stop the camera before switching to postwork. Recording and media export need FFmpeg; the current live recorder uses NVIDIA NVENC.
- Face Lab generation is optional and has a separate large runtime/model setup. See [optional Face Lab](docs/INSTALL.md#optional-face-lab). It is not required for face swapping or virtual output.

Virtual output starts off every time. The output pipeline uses validated processed frames and a safety slate on startup/expired processing, rather than raw camera fallback. This behavior has automated coverage, but this experimental build is not certified for unattended broadcasts.

## Support and limitations

[Installation and troubleshooting](docs/INSTALL.md), [models](docs/MODELS.md), [tester checklist](docs/TESTING.md) and [release provenance](docs/RELEASE.md) describe exactly what was checked and what remains unverified. Output delivery can repeat processed frames; **output cadence is not fresh inference FPS**. Browser/Discord receipt, long sessions and combined Voicemod load need tester acceptance.

Logs, settings, generated files and caches stay under `state/` and are ignored by Git. Logs may contain local file paths or device names; review them before sharing. The source retains the upstream [AGPL-3.0 license](LICENSE); [third-party notices](THIRD_PARTY_NOTICES.md) cover the bundled segmentation asset and separately acquired models/dependencies.
