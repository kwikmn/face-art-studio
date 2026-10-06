# Attribution and relevant licenses

FaceArt Studio is derived from **[hacksider/Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)** through **[kwikmn/Deep-Live-Cam](https://github.com/kwikmn/Deep-Live-Cam)**, based on local commit **`6cf2d087560f47fec79acfbeb6b4ea11be69dc47`**. Its original **GNU Affero General Public License v3** is retained verbatim in [LICENSE](LICENSE). This is a sanitized source snapshot; the experimental repository's historical Git data is not copied. The date, baseline hashes and FaceArt changes are documented in [RELEASE.md](docs/RELEASE.md). No replacement license is assigned to upstream code.

The bundled MediaPipe ONNX background model comes from [royshil/obs-backgroundremoval, revision ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808](https://github.com/royshil/obs-backgroundremoval/blob/ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808/data/models/mediapipe.onnx). Its adjacent `.license` file credits **Google LLC (2021), Katsuya Hyodo (2021), Roy Shilkrot (2021–2026), and Kaito Udagawa (2023–2026)** and identifies **Apache-2.0**. That attribution, the full Apache-2.0 text and source/hash README are included with the asset. FaceArt's Python compositor is independent of the OBS plugin implementation. The FaceArt icon is retained from the supplied Studio source.

**Separately acquired dependencies/models retain their own terms:**

- [InsightFace](https://pypi.org/project/insightface/0.7.3/): MIT library code; supplied pretrained weights limited to non-commercial research. Swapper/buffalo weights are excluded.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): MIT. [NVIDIA runtime wheels](https://pypi.org/project/nvidia-cudnn-cu12/9.10.2.21/) use NVIDIA's proprietary terms; the setup obtains packages from the vendor on PyPI instead of redistributing copied DLLs.
- [PySide6](https://doc.qt.io/qtforpython-6/licenses.html): Qt/PySide licensing applies to the installed packages (including LGPL/GPL/commercial options). Keep the package notices when redistributing a packaged application.
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam): GPL-2.0; [OBS Studio](https://github.com/obsproject/obs-studio): GPL-2.0. Neither the OBS installer nor its driver is bundled here. The installed driver is required for output.
- [RVM](https://github.com/PeterL1n/RobustVideoMatting): GPL-3.0 author repository. [MODNet](https://github.com/ZHKKKe/MODNet): Apache-2.0 author repository, with an author-endorsed community ONNX conversion. Their weights are linked rather than bundled.
- [GPEN](https://github.com/yangxy/GPEN), [GFPGAN](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE) and their ONNX conversions have the restrictions/provenance limits described in [MODELS.md](docs/MODELS.md). Rights for commercial use of the mirrored conversions have not been established here; their binaries are excluded.
- Optional [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B): consult its model card/license at the selected revision. This build does not include the snapshot or grant access rights.
- [FFmpeg](https://ffmpeg.org/legal.html): build-dependent LGPL/GPL terms. Obtain a build yourself and retain its notices; no FFmpeg binaries are shipped.

Dependency license files installed with their packages remain authoritative. Source sharing under AGPL does not make restricted model weights unrestricted. This repository gives testers corresponding application source; invitations/visibility are managed separately by its owner.
