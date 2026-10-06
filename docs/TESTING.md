# Tester acceptance checklist

Use permitted source images and a local preview first. Keep personal source images, recordings and settings out of bug reports. Setup/check commands never open capture devices or virtual output.

1. Confirm setup prerequisites; run `Setup-FaceArt.cmd`, place the required models, and run `Check-FaceArt.cmd`. Record Windows/Python, GPU/driver, app commit and the check outcome. A successful setup is not yet browser/Discord acceptance.
2. Launch Studio at 540p. Choose your source face and physical camera. Verify processing starts and the preview shows the selected replacement face. Try losing/reacquiring face detection and stopping/restarting the camera; expired processing should show the slate, without a raw face fallback.
3. Try background off, blur, color, a permitted image, and a slideshow with two local images. Verify crossfades affect the background without crossfading the face/foreground. Compare MediaPipe/RVM/MODNet if installed. Test chair cleanup under MediaPipe and reduce it when hair or arms are clipped.
4. Confirm virtual output is off on launch/restart. Start it explicitly; choose **OBS Virtual Camera** in a receiving app's local preview. Check dimensions, color, background effect, stop/restart and busy-producer errors. OBS's own producer must not run simultaneously.
5. Select Voicemod separately as the receiving microphone. Check sync/load only in a test you choose to perform. Virtual camera transport is video only.
6. If optional GPEN is installed, compare it on/off and note fresh processing performance. If FFmpeg is installed, test a short recording and postwork export using media you are permitted to record. Verify saved output/audio yourself; these actions are not part of automated publication checks.
7. Run a longer session and observe memory/GPU usage, receiver stability and recovery after stopping. Log **fresh processing FPS** separately from virtual output/delivery FPS.

Report reproducible steps, expected/actual behavior, commit SHA, machine/runtime versions and sanitized error text through GitHub Issues. Upload pictures/logs only after reviewing them for faces, usernames, local paths, tokens and device identifiers.

## Automated developer checks

With the base environment installed, from the repository folder:

```powershell
.\venv\Scripts\python.exe -s scripts\run_tests.py
.\venv\Scripts\python.exe -s scripts\check_install.py --packages-only
.\venv\Scripts\python.exe -s scripts\check_install.py --models-only --hash-models
```

`scripts/run_tests.py` runs each test file in its own process because several retained legacy tests replace imported modules. The optional Face Lab batch-generation fixture is skipped when PyTorch is absent; it does not generate through a real model.

`validate_virtual_ui.py` is an offscreen Qt controls fixture and opens no hardware. Its screenshots/JSON go to ignored `state/outputs/`. The earlier Candidate 5 DirectShow synthetic-receipt test is described in `RELEASE.md`; its research recordings/logs and hardware-activating receipt script are intentionally excluded from this release.

No automated check establishes webcam quality, inference performance, browser/Discord integration or broadcast readiness. See [release verification](RELEASE.md).
