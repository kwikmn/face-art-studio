# v0.2 tester release provenance

Prepared **2026-10-06**, based on **FaceArt Studio v0.2 Candidate 5**. The source candidate's frozen manifest was checked against its current files before selecting release content; all entries matched. The upstream/fork base commit is `6cf2d087560f47fec79acfbeb6b4ea11be69dc47`. [candidate5-source.json](candidate5-source.json) records relative paths and hashes of the original selected files; it is a baseline, not a checksum manifest of this final release.

The initial source selection was 99 files, about 1.5 MB. It includes the complete application modules, locale data, unit fixtures, icon, redistributable MediaPipe fallback with notices, runtime entry points and matting metadata. Installation scripts, documentation and CI were then added.

Excluded: all experimental Git history, preserved candidate directories, virtual environments, copied CUDA DLLs, large/restricted weights, Hugging Face/Torch/pip caches, user settings, selected face/background images, recordings, screenshots, personal logs, test-output folders and redundant research notes. No secrets or credential files are intentionally included. The source's stable v0.1 install and existing file transfer were not modified.

## Changes for distribution

- Added setup/start/check launchers with Windows/Python/GPU/Build Tools/runtime checks, a local venv, official PyPI sources, compatible pinned requirements, repeatable setup and local logs. System installers, drivers, credentials and camera/microphone capture are outside setup.
- Studio startup now loads CUDA libraries through `runtime_bootstrap.py` without importing the legacy Deep-Live-Cam UI, TensorFlow or opennsfw2. The protected face-processing/output behavior is retained.
- Required models are acquired manually; missing models return clear placement instructions. Restoration no longer silently downloads missing weights. The retained legacy face analyser also uses the project model pack instead of a user-wide cache.
- Added model size/hash metadata, source attribution, license notices and tester instructions. No alternative license was created for upstream code.
- Reformatted slideshow implementation/UI and related tests to satisfy the existing lint rules, and removed inherited trailing whitespace, without intended behavior changes. Test fixtures run in isolated processes to avoid import-stub leakage; the optional PyTorch generation fixture is skipped if its optional runtime is absent.

The supplied historical requirements requested newer versions than the Candidate 5 runtime. This release replaces them with compatible setup pins, adding cuFFT/nvJitLink wheels used by ONNX Runtime's standard CUDA preloader. Official PyPI metadata was checked for **72 pinned versions** including bootstrap packages: Python 3.10 requirements and all active Windows base dependency constraints matched, with no missing pins identified. This is metadata validation, not a completed fresh pip installation.

## Verification boundaries

- **79 retained unit cases:** 78 passed, 1 optional Face Lab batch-generation fixture skipped because PyTorch is absent from the available test interpreter. Background, live pipeline, virtual output, slideshow, pacing, blending, audio and media fixtures passed. These tests use synthetic data/mocks.
- Source lint, compilation, setup dry run/prerequisite checks, missing-model reporting and relative documentation links were checked. An isolated clean source checkout is used for final verification.
- The runtime checker imported the actual Studio and completed synthetic CUDA inference using the pre-existing interpreter and original CUDA DLL directory (registered for that test process only). The available original runtime emitted an unused cuFFT preload warning; the setup now specifies its official cuFFT/nvJitLink wheels. The offscreen Qt virtual-camera controls fixture also passed without hardware activation. This does not establish a fresh package installation.
- Required/optional ONNX model links and the official InsightFace/RVM releases were verified through read-only HTTP metadata; no new weights were downloaded. MODNet's author-linked Drive page was verified, while its binary was not reacquired. Known-working model hashes identify the original files.
- The original Candidate 5 record reports synthetic DirectShow receipt through OBS Virtual Camera, slate startup/failure behavior, busy-producer rejection and stop/restart/release. Its roughly 30–32 received frames/sec included repeated frames and is **not** a fresh face-swap inference FPS measurement. Original research artifacts are excluded from publication.

**Remaining acceptance:** fresh Windows package installation/build, real webcam face swapping in this packaged source, browser/Discord virtual-camera receipt, long sessions, performance across GPUs, recording/export on a tester's FFmpeg build, optional Face Lab installation and simultaneous Voicemod load. Publication verification does not capture camera/microphone data or enable virtual output. No livestream-readiness claim is made.

## Public sharing considerations

The owner changed the same destination repository to public during preparation; this release follows that authorization and does not change repository visibility or grant tester access. Testers can download source and submit issues without being given write access.

Restricted/unclear model redistribution is addressed by omitting the binaries and documenting exact links/terms. InsightFace's non-commercial research restriction remains; rights for commercial use of mirrored GPEN/GFPGAN conversions are not established here. Public availability of the application source does not grant model rights. A future bundled executable would need its own dependency/model license and notice audit. Keep personal logs, media and future issue attachments out of the repository.
