# Model downloads and placement

Restricted and large weights are not included in this repository. **Downloading a model does not grant broader rights than its author permits.** In particular, InsightFace's pretrained models are for **non-commercial research only**, including manual downloads; see [InsightFace 0.7.3's official package terms](https://pypi.org/project/insightface/0.7.3/). Commercial streaming/use needs appropriate separately licensed models and compatibility work. Keeping the code private or making it public does not change model terms.

Links below were checked on **2026-10-06**. Direct weight/archive links returned HTTP 200 with the expected sizes. MODNet's author-linked Drive page returned HTTP 200; its actual binary was not downloaded again. `downloads.json` records exact expected sizes and SHA-256 values from the known-working Candidate 5 files. These checksums identify that build's files, rather than certify licensing or independently establish provenance.

## Required for live face swapping

1. Download the full-precision **[inswapper_128.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/581e70b61240b7928404c17900437f47cfe94133/inswapper_128.onnx)** (554,253,681 bytes) and place it at **`models/inswapper_128.onnx`**. This is a pinned copy from the **Deep-Live-Cam upstream maintainer's Hugging Face mirror**, not an unrestricted model authored by FaceArt. [InsightFace's original project](https://github.com/deepinsight/insightface) governs its pretrained model rights. Do not substitute `inswapper_128_fp16.onnx`: Candidate 5's protected live path expects the full-precision model.
2. Download **[buffalo_l.zip from InsightFace's official release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip)**. The release archive is 288,621,354 bytes. Extract its five `.onnx` files directly into **`models/buffalo_l/`**, without a second `buffalo_l` folder. The live path needs `det_10g.onnx` and `w600k_r50.onnx`; keep the remaining pack members for analysis and postwork.

Expected layout relative to the folder containing `Start-FaceArt.cmd`:

```text
models/
  inswapper_128.onnx
  buffalo_l/
    det_10g.onnx
    w600k_r50.onnx
    2d106det.onnx
    1k3d68.onnx
    genderage.onnx
```

Run `Check-FaceArt.cmd` after placement. For a complete checksum check:

```powershell
.\venv\Scripts\python.exe -s scripts\check_install.py --hash-models
```

The source uses the project model directory directly, so there is no requirement to put your files into a user-wide `.insightface` cache. Do not send replacement-face images with a setup report. Choose your own permitted image inside Studio after installation.

## Optional enhancement/restoration

These links are pinned **Deep-Live-Cam maintainer mirror copies of ONNX conversions**. Original authors and conversion sources are linked separately; mirror hosting is not an extra license grant.

| Feature | Download | Exact destination |
| --- | --- | --- |
| Live GPEN and fast postwork | [GPEN-BFR-256.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/581e70b61240b7928404c17900437f47cfe94133/GPEN-BFR-256.onnx) | `models/GPEN-BFR-256.onnx` |
| Detailed postwork | [GPEN-BFR-512.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/581e70b61240b7928404c17900437f47cfe94133/GPEN-BFR-512.onnx) | `models/GPEN-BFR-512.onnx` |
| GFPGAN restoration | [GFPGANv1.4.onnx](https://huggingface.co/hacksider/deep-live-cam/resolve/581e70b61240b7928404c17900437f47cfe94133/GFPGANv1.4.onnx) | `models/GFPGANv1.4.onnx` |

GPEN originates with [yangxy/GPEN](https://github.com/yangxy/GPEN); [Face-Upscalers-ONNX](https://github.com/harisreedhar/Face-Upscalers-ONNX) provides ONNX conversion releases. The original GPEN repository does not provide a clear top-level license grant for every mirrored conversion; **commercial and redistribution rights for these files have not been established here**. Obtain permission as needed. No GPEN weights are redistributed by this repo.

GFPGAN originates with [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN). Its [license](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE) describes Apache-2.0 and separately listed third-party components, including restrictive terms. Do not assume an exported ONNX file removes those terms. GFPGAN weights are also not redistributed here.

This release asks for missing restoration models explicitly instead of silently downloading them on first use.

## Background masks

The **MediaPipe fallback is included** at `modules/studio/assets/background/mediapipe.onnx`, with Apache-2.0 text and adjacent copyright attribution. Its exact upstream source revision and hash are in the adjacent README. Nothing else is needed for this backend.

For optional backends:

| Backend | Author/source | Exact destination |
| --- | --- | --- |
| RVM | [Official RVM ONNX download](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx); [author repository, GPL-3.0](https://github.com/PeterL1n/RobustVideoMatting) | `models/matting/rvm_mobilenetv3_fp32.onnx` |
| MODNet | [Author-endorsed community ONNX download page](https://drive.google.com/file/d/1cgycTQlYXpTh26gB9FTnthE7AvruV8hd/view); [author's ONNX instructions](https://github.com/ZHKKKe/MODNet/blob/master/onnx/README.md) and [Apache-2.0 project](https://github.com/ZHKKKe/MODNet) | `models/matting/modnet.onnx` |

For MODNet, use Drive's download control to get the actual ONNX file and rename it **`modnet.onnx`**. Saving the displayed HTML page will not work. Keep the included `models/matting/manifest.json`; the runtime verifies its known-working RVM and MODNet hashes. Select MediaPipe when optional files are absent.

## Optional Face Lab model

Face generation uses **[black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)**, pinned to revision **`e7b7dc27f91deacad38e78976d1f2b499d76a294`**. Review that provider model card and license; the 4B card identifies Apache-2.0. Other FLUX variants can have different terms.

Download the model snapshot at that revision into **`models/FLUX.2-klein-4B/`**. Preserve its relative directories: `model_index.json`, `scheduler/`, `text_encoder/`, `tokenizer/`, `transformer/`, `vae/`, `LICENSE.md` and `README.md`. Use the official Hugging Face site/CLI from your own account if needed. The repo does not contain access tokens or these large weights. Install the [optional runtime](INSTALL.md#optional-face-lab) before trying generation; normal face swapping does not need it.

Avoid mixing weights from different revisions. If a provider replaces/removes a file or access conditions change, stop and follow the author instructions; do not use arbitrary mirrors.
