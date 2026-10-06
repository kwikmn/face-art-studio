"""Single-face FP32 processor for the protected webcam.

Uses dedicated sessions with bounded CPU pools. No enhancer, original-mouth
paste, partial opacity, or raw-frame fallback is reachable from this path.
"""
from pathlib import Path
import time
from collections import defaultdict, deque
from modules.telemetry import profile_options, register

import cv2
import numpy as np

from modules.live_pipeline import ProcessedFrame, fit_frame
from modules.paths import MODELS_DIR
from modules.face_blending import face_mask


def validate_face(face, shape, min_score=0.65):
    score = float(getattr(face, 'det_score', 0))
    if face is None or not np.isfinite(score) or score < min_score:
        raise ValueError("Face confidence too low")
    box = np.asarray(face.bbox)
    points = np.asarray(face.kps)
    h, w = shape[:2]
    if (box.shape != (4,) or points.shape != (5, 2) or
            not np.isfinite(box).all() or not np.isfinite(points).all()):
        raise ValueError("Invalid face geometry")
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if min(bw, bh) < 40:
        raise ValueError("Face too small")
    # Detector boxes estimate a forehead/chin envelope, not exact face pixels.
    # Permit <=5% envelope overrun, only with >=95% visible area and all five
    # alignment landmarks comfortably inside the image. Never clamp landmarks.
    visible_w = max(0, min(x2, w) - max(x1, 0))
    visible_h = max(0, min(y2, h) - max(y1, 0))
    if (x1 < -.05 * bw or y1 < -.05 * bh or
            x2 > w + .05 * bw or y2 > h + .05 * bh or
            visible_w * visible_h < .95 * bw * bh):
        raise ValueError("Keep the entire face inside the picture")
    margin = max(2, .03 * min(bw, bh))
    if ((points[:, 0] < margin).any() or (points[:, 0] >= w-margin).any() or
            (points[:, 1] < margin).any() or (points[:, 1] >= h-margin).any()):
        raise ValueError("Keep facial landmarks inside the picture")
    if ((points[:, 0] < x1).any() or (points[:, 0] > x2).any() or
            (points[:, 1] < y1).any() or (points[:, 1] > y2).any()):
        raise ValueError("Face landmarks outside face")


def paste_checked(frame, fake, matrix, alpha):
    """Validate inference output and blend on CPU, with no silent return path."""
    size = alpha.shape[0]
    if (not isinstance(fake, np.ndarray) or fake.dtype != np.uint8 or
            fake.shape != (size, size, 3)):
        raise ValueError("Invalid face-swap image")
    matrix = np.asarray(matrix)
    if (matrix.shape != (2, 3) or not np.isfinite(matrix).all() or
            abs(np.linalg.det(matrix[:, :2])) < 1e-8):
        raise ValueError("Invalid face-swap transform")
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    corners = corners @ inverse[:, :2].T + inverse[:, 2]
    h, w = frame.shape[:2]
    x1, y1 = np.maximum(np.floor(corners.min(axis=0)).astype(int) - 2, (0, 0))
    x2, y2 = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 3, (w, h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Face-swap region outside picture")
    inverse[:, 2] -= (x1, y1)
    extent = (int(x2-x1), int(y2-y1))
    mask = cv2.warpAffine(alpha, inverse, extent, borderValue=0)
    if np.count_nonzero(mask > 128) < 100:
        raise ValueError("Insufficient face replacement coverage")
    face = cv2.warpAffine(fake, inverse, extent, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.merge([mask, mask, mask])
    result = frame.copy()
    result[y1:y2, x1:x2] = cv2.add(
        cv2.multiply(face, mask, scale=1/255),
        cv2.multiply(frame[y1:y2, x1:x2], 255-mask, scale=1/255))
    return result


class CudaReplaySession:
    """Reuse GPU buffers without graph capture while models load concurrently."""
    def __init__(self, original, path):
        import onnxruntime as ort
        self.original = original
        self.bindings = None
        self.inputs = {}
        self.failed = False
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
        opts.add_session_config_entry('session.inter_op.allow_spinning', '0')
        profile_options(opts,'replay-'+Path(path).stem)
        # Graph capture conflicts with CUDA model uploads on the background
        # GPEN/source loader (CUDA error 900). Keep IO binding, disable capture.
        self.session = ort.InferenceSession(str(path), sess_options=opts,
            providers=[('CUDAExecutionProvider', {'enable_cuda_graph': '0'})])
        register(self.session,'replay-'+Path(path).stem)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run(self, output_names, inputs, **kwargs):
        if not self.failed:
            try:
                import onnxruntime as ort
                # update_inplace copies linear memory, ignoring NumPy strides.
                # A transposed HWC -> NCHW view must be packed before GPU upload.
                inputs = {name: np.ascontiguousarray(value) for name, value in inputs.items()}
                if self.bindings is None:
                    self.bindings = self.session.io_binding()
                    for name, value in inputs.items():
                        self.inputs[name] = ort.OrtValue.ortvalue_from_numpy(value, 'cuda', 0)
                        self.bindings.bind_ortvalue_input(name, self.inputs[name])
                    for output in self.session.get_outputs():
                        self.bindings.bind_output(output.name, 'cuda', 0)
                else:
                    for name, value in inputs.items():
                        self.inputs[name].update_inplace(value)
                self.session.run_with_iobinding(self.bindings)
                return self.bindings.copy_outputs_to_cpu()
            except Exception as error:
                # The original, non-graph inference session remains available.
                print(f'[protected-webcam] CUDA replay unavailable: {error}', flush=True)
                self.failed = True
                self.bindings = None
                self.inputs = {}
                self.session = None
        return self.original.run(output_names, inputs, **kwargs)


class LiveFaceProcessor:
    def __init__(self, source_path, providers=None, width=1280, height=720, mirror=False):
        self.source_path = str(source_path)
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.width, self.height, self.mirror = width, height, mirror
        self.provider_report = {}
        self.timings = {}
        self.stage_samples=defaultdict(lambda:deque(maxlen=900))
        self.background_settings = None
        self.background_effect = None
        self.background_error = None
        self._live_enhancement = None
        self.enhancement_error = None
        self._jaw_feather = .6
        self._mask_key = (128, .6)
        self.alpha = face_mask(*self._mask_key)

    def set_jaw_feathering(self, strength):
        if not np.isfinite(strength):
            raise ValueError("Invalid jaw feathering")
        self._jaw_feather = max(0., min(1., strength))

    def _mask_for(self, size):
        # Only the processing worker updates the cached array. UI changes are
        # applied between frames, and existing frames retain their own mask.
        key = (size, self._jaw_feather)
        if key != self._mask_key:
            self.alpha = face_mask(*key)
            self._mask_key = key
        return self.alpha

    def _load(self, path):
        import onnxruntime as ort
        from insightface.model_zoo.model_zoo import ModelRouter
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing model: {path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        profile_options(options,Path(path).stem)
        # ModelRouter accepts sess_options; insightface.get_model drops it in 0.7.3.
        model = ModelRouter(str(path)).get_model(providers=self.providers, sess_options=options)
        if model is None:
            raise RuntimeError(f"Unsupported model: {path}")
        actual = model.session.get_providers()
        register(model.session,Path(path).stem)
        self.provider_report[Path(path).name] = actual
        requested = [p if isinstance(p, str) else p[0] for p in self.providers]
        if 'CUDAExecutionProvider' in requested and 'CUDAExecutionProvider' not in actual:
            raise RuntimeError("CUDA unavailable: refusing silent CPU fallback")
        return model

    def prepare(self):
        cv2.setNumThreads(1)
        pack = Path(MODELS_DIR) / 'buffalo_l'
        self.detector = self._load(pack / 'det_10g.onnx')
        self.detector.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)
        recognizer = self._load(pack / 'w600k_r50.onnx')
        recognizer.prepare(ctx_id=0)
        self.swapper = self._load(Path(MODELS_DIR) / 'inswapper_128.onnx')
        self.source = self._read_source(self.source_path, self.detector, recognizer)
        size = self.swapper.input_size[0]
        self._mask_for(size)
        # Source recognition is only needed once; release its session.
        del recognizer
        if 'CUDAExecutionProvider' in self.swapper.session.get_providers():
            try:
                self.swapper.session = CudaReplaySession(
                    self.swapper.session, Path(MODELS_DIR) / 'inswapper_128.onnx')
            except Exception as error:
                print(f'[protected-webcam] CUDA replay setup skipped: {error}', flush=True)

    @staticmethod
    def _read_source(path, detector, recognizer):
        from insightface.app.common import Face
        from modules import imread_unicode
        source = imread_unicode(path)
        if source is None:
            raise ValueError("Cannot read replacement-face image")
        boxes, points = detector.detect(source, max_num=0)
        if len(boxes) != 1 or points is None:
            raise ValueError("Replacement image must contain exactly one detected face")
        face = Face(bbox=boxes[0, :4], kps=points[0], det_score=boxes[0, 4])
        validate_face(face, source.shape)
        recognizer.get(source, face)
        embedding = face.normed_embedding
        if embedding is None or not np.isfinite(embedding).all() or np.linalg.norm(embedding) < 0.9:
            raise ValueError("Invalid replacement-face embedding")
        return face

    def prepare_source_image(self, path):
        # Separate temporary sessions: never use the live detector concurrently.
        loader = LiveFaceProcessor(path, self.providers)
        pack = Path(MODELS_DIR) / 'buffalo_l'
        detector = loader._load(pack / 'det_10g.onnx')
        detector.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)
        recognizer = loader._load(pack / 'w600k_r50.onnx')
        recognizer.prepare(ctx_id=0)
        return self._read_source(path, detector, recognizer)

    def activate_source(self, face, path):
        # Called only by the live processing worker, between frames.
        self.source, self.source_path = face, str(path)

    def set_background(self, settings):
        # An immutable settings snapshot is consumed between processed frames.
        self.background_settings = settings

    def _apply_background(self, frame, face_box=None, face_region=None):
        settings = self.background_settings
        if settings is None or settings.mode == "off":
            if self.background_effect is not None:
                self.background_effect.reset()
            self.background_error = None
            self.timings["background_ms"] = 0.0
            return frame
        started = time.perf_counter()
        try:
            if self.background_effect is None:
                from modules.studio.background import BackgroundEffect
                self.background_effect = BackgroundEffect()
            self.background_effect.slideshow = getattr(self,'slideshow',None)
            result = self.background_effect.apply(frame, settings, face_box, face_region)
            self.background_error = None
            return result
        except Exception as error:
            self.background_error = str(error)
            # The pipeline clears output on an exception; never return the room
            # unmasked just because the requested background effect failed.
            raise RuntimeError(f"Background effect: {error}") from error
        finally:
            self.timings["background_ms"] = (time.perf_counter() - started) * 1000

    def set_live_enhancement(self, enhancer=None, strength=.6):
        self._live_enhancement = (enhancer, max(0., min(1., strength))) if enhancer else None
        self.enhancement_error = None

    def stage_metrics(self):
        return {key:{'samples':len(values),'warmup_excluded':min(5,len(values)), 'p50_ms':float(np.percentile(list(values)[5:],50)) if len(values)>5 else None,'p95_ms':float(np.percentile(list(values)[5:],95)) if len(values)>5 else None} for key,values in self.stage_samples.items()}

    def process(self, frame):
        from insightface.app.common import Face
        started = time.perf_counter()
        frame = fit_frame(frame, self.width, self.height)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        prepared=time.perf_counter()
        stages = {} if getattr(self, 'capture_stages', False) else None
        if stages is not None:
            stages['prepared_raw'] = frame.copy()
        boxes, points = self.detector.detect(frame, max_num=0)
        detected = time.perf_counter()
        if len(boxes) != 1 or points is None:
            return ProcessedFrame(None, 'No face detected' if len(boxes) == 0 else 'Multiple faces detected')
        target = Face(bbox=boxes[0, :4], kps=points[0], det_score=boxes[0, 4])
        validate_face(target, frame.shape)
        fake, matrix = self.swapper.get(frame, target, self.source, paste_back=False)
        swapped = time.perf_counter()
        alpha = self._mask_for(fake.shape[0])
        if stages is not None:
            stages['swapped'] = paste_checked(frame, fake, matrix, alpha)
        enhancement = self._live_enhancement
        if enhancement is not None:
            try:
                fake, matrix, alpha = enhancement[0].apply(fake, matrix, alpha, enhancement[1])
                self.enhancement_error = None
            except Exception as error:
                # The fallback is the already-swapped crop, never the original face.
                self.enhancement_error = str(error)
        enhanced = time.perf_counter()
        result = paste_checked(frame, fake, matrix, alpha)
        if stages is not None:
            stages['enhanced'] = result.copy()
            self.last_stages = stages
        self.timings = {'prepare_frame_ms':(prepared-started)*1000,'detect_ms': (detected-prepared)*1000,
                        'swap_ms': (swapped-detected)*1000,
                        'enhance_ms': (enhanced-swapped)*1000,
                        'paste_ms': (time.perf_counter()-enhanced)*1000}
        output=self._apply_background(result,target.bbox,(matrix,alpha))
        if self.background_effect is not None and self.background_effect.matting is not None:
            self.timings.update(self.background_effect.matting.timings)
        for key,value in self.timings.items():
            self.stage_samples[key].append(value)
        return ProcessedFrame(output)
