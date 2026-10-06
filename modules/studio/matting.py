"""Author-distributed ONNX matting. Never replaces protected foreground pixels."""
from pathlib import Path
import json
import hashlib
import numpy as np
import cv2
import os
import time
from modules.telemetry import profile_options, register

MODEL_DIR = Path(__file__).resolve().parents[2] / 'models' / 'matting'
MODEL_FILES = {'rvm': 'rvm_mobilenetv3_fp32.onnx', 'modnet': 'modnet.onnx'}


class MattingBackend:
    def __init__(self, name, provider='cpu', ratio=0.375):
        import onnxruntime as ort
        self.name, self.provider, self.ratio = name, provider, ratio
        path = MODEL_DIR / MODEL_FILES[name]
        if not path.is_file():
            raise FileNotFoundError(f'{name} model missing: {path}. Select MediaPipe fallback.')
        manifest = MODEL_DIR / 'manifest.json'
        if manifest.is_file():
            expected = json.loads(manifest.read_text())['models'][name]['sha256']
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'{name} model checksum mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        profile_options(options,name)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if provider == 'cuda' else ['CPUExecutionProvider']
        self.session = register(ort.InferenceSession(str(path), sess_options=options, providers=providers),name)
        self.gpu_state = name=='rvm' and provider=='cuda' and os.environ.get('FACEART_RVM_REFERENCE')!='1'
        self.timings = {}
        self.providers = self.session.get_providers()
        if provider == 'cuda' and 'CUDAExecutionProvider' not in self.providers:
            raise RuntimeError('CUDA unavailable; explicitly select CPU or MediaPipe fallback')
        self.reset()

    def reset(self):
        self.shape = None
        self.rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]
        self.gpu_rec = None
        self.gpu_src = None
        self.gpu_ratio = None

    def mask(self, frame):
        started=time.perf_counter()
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Matting expects uint8 BGR')
        if self.shape != frame.shape:
            self.reset()
            self.shape = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.name == 'rvm':
            src = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255
            prepared=time.perf_counter()
            if self.gpu_state:
                import onnxruntime as ort
                if self.gpu_rec is None:
                    self.gpu_rec=[ort.OrtValue.ortvalue_from_numpy(value,'cuda',0) for value in self.rec]
                    self.gpu_src=ort.OrtValue.ortvalue_from_numpy(src,'cuda',0)
                    self.gpu_ratio=ort.OrtValue.ortvalue_from_numpy(np.array([self.ratio],np.float32),'cuda',0)
                else:
                    self.gpu_src.update_inplace(src)
                io=self.session.io_binding()
                io.bind_ortvalue_input('src',self.gpu_src)
                io.bind_ortvalue_input('downsample_ratio',self.gpu_ratio)
                for i,value in enumerate(self.gpu_rec):
                    io.bind_ortvalue_input(f'r{i+1}i',value)
                for name in ('pha','r1o','r2o','r3o','r4o'):
                    io.bind_output(name,'cuda',0)
                self.session.run_with_iobinding(io)
                outputs=io.get_outputs()
                inferred=time.perf_counter()
                alpha=outputs[0].numpy()
                self.gpu_rec=outputs[1:]
            else:
                feeds = {'src': src, 'downsample_ratio': np.array([self.ratio], np.float32)}
                feeds.update({f'r{i+1}i': value for i, value in enumerate(self.rec)})
                _, alpha, *rec = self.session.run(None, feeds)
                self.rec = rec
                inferred=time.perf_counter()
        else:
            h, w = rgb.shape[:2]
            scale = 512 / min(h, w) if max(h, w) < 512 or min(h, w) > 512 else 1
            rh, rw = max(32, int(h * scale) // 32 * 32), max(32, int(w * scale) // 32 * 32)
            inp = self.session.get_inputs()[0]
            if isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
                rh, rw = inp.shape[2:]
            normalized = (rgb.astype(np.float32) - 127.5) / 127.5
            small = cv2.resize(normalized, (rw, rh), interpolation=cv2.INTER_AREA)
            src = np.ascontiguousarray(small.transpose(2, 0, 1)[None], dtype=np.float32)
            prepared=time.perf_counter()
            alpha = self.session.run(None, {inp.name: src})[-1]
            inferred=time.perf_counter()
        alpha = np.asarray(alpha).squeeze()
        if alpha.ndim != 2 or not np.isfinite(alpha).all():
            raise ValueError('Invalid matting alpha')
        alpha=np.clip(alpha,0,1)
        if alpha.shape != frame.shape[:2]:
            alpha=cv2.resize(alpha,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_LINEAR)
        self.timings={'matting_prepare_ms':(prepared-started)*1000,'matting_execute_transfer_ms':(inferred-prepared)*1000,'matting_alpha_readback_post_ms':(time.perf_counter()-inferred)*1000}
        return alpha
