"""Person segmentation and background compositing on protected BGR frames.

Independent implementation using the Apache-2.0 MediaPipe landscape weights.
All model, image and temporal state belongs to the live processing worker.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

from modules import imread_unicode

MODEL_PATH = Path(__file__).with_name("assets") / "background" / "mediapipe.onnx"
MODEL_SHA256 = "7f785cf032261a07af7b845f891cab30da3f0757c7b362310e089e3aa8e8860a"


@dataclass(frozen=True)
class BackgroundSettings:
    mode: str = "off"
    color: str = "#20343b"
    image: str = ""
    blur: int = 22
    softness: int = 5
    smoothing: int = 20
    chair_cleanup: int = 50
    tightness: int = 25
    backend: str = "mediapipe"
    provider: str = "cpu"
    downsample_ratio: float = 0.375
    preserve_alpha: bool = True

    def __post_init__(self):
        if self.backend not in ('mediapipe', 'rvm', 'modnet') or self.provider not in ('cpu', 'cuda'):
            raise ValueError('Unknown matting backend/provider')
        if not 0.1 <= self.downsample_ratio <= 1:
            raise ValueError('Invalid RVM downsample ratio')
        if self.mode not in ("off", "blur", "color", "image", "slideshow"):
            raise ValueError("Unknown background mode")
        if len(self.color) != 7 or not self.color.startswith("#"):
            raise ValueError("Invalid background color")
        int(self.color[1:], 16)
        for name, low, high in (
            ("blur", 1, 50),
            ("softness", 0, 20),
            ("smoothing", 0, 60),
            ("chair_cleanup", 0, 100),
            ("tightness", 0, 100),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid background {name}")


class BackgroundEffect:
    def __init__(self):
        self.session = None
        self.load_error = None
        self.previous = None
        self.previous_key = None
        self.image_key = None
        self.image_frame = None
        self.matting = None
        self.matting_key = None
        self.last_alpha = None
        self.slideshow = None

    def reset(self):
        self.previous = self.previous_key = self.last_alpha = None
        if self.matting is not None:
            self.matting.reset()

    def _load(self):
        if self.load_error:
            raise RuntimeError(self.load_error)
        if self.session is not None:
            return
        try:
            import onnxruntime as ort

            if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
                raise RuntimeError("Background model is missing or damaged")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            self.session = ort.InferenceSession(
                str(MODEL_PATH),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
        except Exception as error:
            self.load_error = f"Background model unavailable: {error}"
            raise RuntimeError(self.load_error) from error

    def _mask(self, frame):
        self._load()
        small = cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(rgb[None], dtype=np.float32) / 255.0
        output = self.session.run(None, {self.input_name: blob})[0]
        if output.shape != (1, 144, 256, 2) or not np.isfinite(output).all():
            raise RuntimeError("Invalid background segmentation output")
        return np.clip(output[0, :, :, 1], 0, 1)

    def _image(self, path, width, height):
        key = (path, width, height)
        if key != self.image_key:
            image = imread_unicode(path) if path else None
            if image is None:
                raise RuntimeError("Choose an available background image")
            scale = max(width / image.shape[1], height / image.shape[0])
            resized = cv2.resize(
                image,
                (
                    max(width, round(image.shape[1] * scale)),
                    max(height, round(image.shape[0] * scale)),
                ),
            )
            y, x = (resized.shape[0] - height) // 2, (resized.shape[1] - width) // 2
            self.image_frame = np.ascontiguousarray(
                resized[y : y + height, x : x + width]
            )
            self.image_key = key
        return self.image_frame

    def _refine_mask(
        self, mask, settings, frame_shape, face_box=None, face_region=None
    ):
        """Use geometry as a confidence prior, never a hard body-shaped crop."""
        strength = settings.chair_cleanup / 100.0
        tightness = settings.tightness / 100.0
        midpoint = np.full(mask.shape, 0.5 + 0.2 * tightness, np.float32)
        if strength and face_box is not None:
            box = np.asarray(face_box, dtype=np.float32)
            if box.shape == (4,) and np.isfinite(box).all():
                x1, y1, x2, y2 = box
                fw, fh = x2 - x1, y2 - y1
                if fw > 0 and fh > 0:
                    yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]].astype(
                        np.float32
                    )
                    xx *= frame_shape[1] / mask.shape[1]
                    yy *= frame_shape[0] / mask.shape[0]
                    # Generous hair/ear allowance, widening below the chin to
                    # make room for shoulders. It only raises the confidence
                    # required outside this region; confident hands can survive.
                    shoulder = np.clip((yy - (y2 - 0.05 * fh)) / (0.7 * fh), 0, 1)
                    radius = fw * (0.65 + 0.85 * shoulder)
                    side = np.clip(
                        (np.abs(xx - (x1 + x2) / 2) / radius - 0.85) / 0.45, 0, 1
                    )
                    above = np.clip(((y1 - 0.3 * fh) - yy) / (0.2 * fh), 0, 1)
                    outside = np.maximum(side, above)
                    midpoint += 0.25 * strength * outside
        # Increase foreground certainty and narrow ambiguous translucent edges.
        half_width = 0.25 - 0.15 * tightness
        result = np.clip((mask - (midpoint - half_width)) / (2 * half_width), 0, 1)
        result = result * result * (3 - 2 * result)
        if settings.softness:
            result = cv2.GaussianBlur(result, (0, 0), settings.softness / 5.0)
        if strength and face_region is not None:
            matrix, alpha = face_region
            inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32))
            inverse[0] *= mask.shape[1] / frame_shape[1]
            inverse[1] *= mask.shape[0] / frame_shape[0]
            # Reuse the actual face replacement mask. Its opaque core must not
            # become transparent while ambiguous chair pixels are suppressed.
            core = (np.asarray(alpha) >= 250).astype(np.float32)
            protected = cv2.warpAffine(core, inverse, (mask.shape[1], mask.shape[0]))
            result = np.maximum(result, protected)
        return result

    def apply(self, frame, settings, face_box=None, face_region=None):
        if settings.mode == "off":
            self.reset()
            return frame
        height, width = frame.shape[:2]
        key = (height, width, settings)
        backend_key = (settings.backend, settings.provider, settings.downsample_ratio)
        if backend_key != self.matting_key:
            self.reset()
            self.matting = None
            if settings.backend != 'mediapipe':
                from modules.studio.matting import MattingBackend
                self.matting = MattingBackend(*backend_key)
            self.matting_key = backend_key
        if self.previous_key != key:
            self.reset()
        mask = self.matting.mask(frame) if self.matting is not None else self._mask(frame)
        if self.previous_key == key and self.previous is not None:
            old_weight = settings.smoothing / 100.0
            mask = cv2.addWeighted(mask, 1 - old_weight, self.previous, old_weight, 0)
        reference_copies=os.environ.get('FACEART_COPY_REFERENCE')=='1'
        self.previous = mask.copy() if reference_copies else mask
        self.previous_key = key
        if settings.backend == 'mediapipe' or not settings.preserve_alpha:
            mask = self._refine_mask(mask, settings, frame.shape, face_box, face_region)
        if mask.shape != (height,width) or reference_copies:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        self.last_alpha = mask.copy() if reference_copies else mask
        if settings.mode == "slideshow":
            background = self.slideshow.frame(width,height) if self.slideshow else None
            if background is None:
                # Safe color while loading/empty: never expose the original room.
                rgb=tuple(int(settings.color[i:i+2],16) for i in (1,3,5))
                color_key=('slideshow-fallback',settings.color,width,height)
                if self.image_key!=color_key:
                    self.image_frame=np.full_like(frame,rgb[::-1]);self.image_key=color_key
                background=self.image_frame
        elif settings.mode == "color":
            rgb = tuple(int(settings.color[i : i + 2], 16) for i in (1, 3, 5))
            color_key = ("color", settings.color, width, height)
            if self.image_key != color_key:
                self.image_frame = np.full_like(frame, rgb[::-1])
                self.image_key = color_key
            background = self.image_frame
        elif settings.mode == "image":
            background = self._image(settings.image, width, height)
        else:
            # Blur at reduced resolution to bound the cost on 720p frames.
            small = cv2.resize(
                frame,
                (max(1, width // 4), max(1, height // 4)),
                interpolation=cv2.INTER_AREA,
            )
            blurred = cv2.GaussianBlur(small, (0, 0), settings.blur / 4.0)
            background = cv2.resize(
                blurred, (width, height), interpolation=cv2.INTER_LINEAR
            )
        # OpenCV blends uint8 images using single-channel float weights without
        # the large temporary float RGB arrays needed by NumPy broadcasting.
        return cv2.blendLinear(frame, background, mask, 1.0 - mask)
