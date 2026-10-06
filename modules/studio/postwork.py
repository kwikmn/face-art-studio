"""Offline restoration jobs; a new file is exported and source audio is retained."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import threading
import uuid

import cv2
import numpy as np

from modules.paths import MODELS_DIR
from modules.studio.recording import executable, HIDDEN, probe

REVISION = "581e70b61240b7928404c17900437f47cfe94133"
MODELS = {
    "GPEN 256 · fast": ("GPEN-BFR-256.onnx", 256),
    "GPEN 512 · detailed": ("GPEN-BFR-512.onnx", 512),
    "GFPGAN 1.4 · restoration": ("GFPGANv1.4.onnx", 512),
}


def obtain_model(name, progress, cancel):
    filename, _ = MODELS[name]
    path = Path(MODELS_DIR) / filename
    if path.is_file():
        return path
    raise FileNotFoundError(f'Missing {filename}. Download it manually using docs/MODELS.md, place it in {MODELS_DIR}, then retry. Model downloads are separate from setup.')


class Restorer:
    def __init__(self, model, strength, progress, cancel):
        import onnxruntime as ort
        from modules.live_face import LiveFaceProcessor

        self.strength = strength
        self.size = MODELS[model][1]
        path = obtain_model(model, progress, cancel)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        loader = LiveFaceProcessor("")
        self.detector = loader._load(
            Path(MODELS_DIR) / "buffalo_l/det_10g.onnx"
        )
        self.detector.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)

    def process(self, frame):
        from modules.processors.frame._onnx_enhancer import enhance_face_onnx

        boxes, points = self.detector.detect(frame, max_num=0)
        enhanced = frame.copy()
        if points is not None:
            for box, landmarks in zip(boxes, points):
                face = SimpleNamespace(bbox=box[:4], kps=landmarks)
                enhanced = enhance_face_onnx(enhanced, face, self.session, self.size)
        return cv2.addWeighted(enhanced, self.strength, frame, 1 - self.strength, 0)


def export_media(
    source, destination, model, strength, progress, cancel=None, swap_source=None
):
    cancel = cancel or threading.Event()
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("Choose a new output filename; the original is preserved")
    destination.parent.mkdir(parents=True, exist_ok=True)
    restorer = Restorer(model, strength, progress, cancel) if model else None
    swapper = None

    def transform(frame):
        nonlocal swapper
        if swap_source:
            if swapper is None:
                from modules.live_face import LiveFaceProcessor

                swapper = LiveFaceProcessor(
                    swap_source, width=frame.shape[1], height=frame.shape[0]
                )
                swapper.prepare()
            result = swapper.process(frame)
            if result.image is None:
                raise ValueError(f"Cannot swap this frame: {result.reason}")
            frame = result.image
        return restorer.process(frame) if restorer else frame

    if source.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        from modules import imread_unicode

        frame = imread_unicode(str(source))
        if frame is None:
            raise ValueError("Cannot read image")
        frame = transform(frame)
        if cancel.is_set():
            raise InterruptedError("Cancelled")
        ok, data = cv2.imencode(destination.suffix, frame)
        if not ok:
            raise RuntimeError("Image encoding failed")
        data.tofile(str(destination))
        progress(100, "Image exported")
        return destination
    info = probe(source)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = map(int, video["avg_frame_rate"].split("/"))
    fps = num / den if den and num else 30
    width, height = video["width"], video["height"]
    total = max(1, round(float(info["format"].get("duration", 1)) * fps))
    temporary = destination.with_name(
        "." + destination.stem + "-" + uuid.uuid4().hex + ".mkv"
    )
    log_path = destination.with_suffix(".postwork.log")
    capture = cv2.VideoCapture(str(source))
    encoder = None
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video")
        with log_path.open("wb") as log:
            encoder = subprocess.Popen(
                [
                    executable("ffmpeg"),
                    "-v",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p4",
                    "-cq",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    str(temporary),
                ],
                stdin=subprocess.PIPE,
                stderr=log,
                stdout=subprocess.DEVNULL,
                creationflags=HIDDEN,
            )
            index = 0
            while True:
                if cancel.is_set():
                    raise InterruptedError("Cancelled")
                ok, frame = capture.read()
                if not ok:
                    break
                frame = transform(frame)
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                index += 1
                progress(
                    min(99, round(index / total * 100)),
                    f"Processing frame {index} / ~{total}",
                )
            encoder.stdin.close()
            if encoder.wait(30) or not index:
                raise RuntimeError("Video export failed; see the postwork log")
            result = subprocess.run(
                [
                    executable("ffmpeg"),
                    "-v",
                    "error",
                    "-n",
                    "-i",
                    str(temporary),
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a?",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "160k",
                    str(destination),
                ],
                stdout=subprocess.DEVNULL,
                stderr=log,
                creationflags=HIDDEN,
                timeout=300,
            )
            if result.returncode:
                raise RuntimeError("Audio muxing failed; see the postwork log")
        progress(100, "Video exported with source audio")
        return destination
    finally:
        capture.release()
        if encoder and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if temporary.exists():
            temporary.unlink()
