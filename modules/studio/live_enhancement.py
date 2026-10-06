"""Optional live restoration of the swapped crop; never uses original face pixels."""

import threading
from pathlib import Path

import cv2
import onnxruntime as ort
import numpy as np

from modules.live_face import CudaReplaySession
from modules.studio.postwork import obtain_model
from modules.telemetry import profile_options, register


class LiveGPEN:
    def __init__(self):
        path = obtain_model("GPEN 256 · fast", lambda *_: None, threading.Event())
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        profile_options(options,'GPEN256')
        self.session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        register(self.session,'GPEN256')
        if "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(
                "Live GPEN needs CUDA on this build; use offline postwork instead"
            )
        self.input_name = self.session.get_inputs()[0].name
        try:
            self.session = CudaReplaySession(self.session, Path(path))
        except Exception:
            pass
        # Warm kernels and allocate GPU buffers before attaching to live video.
        self.session.run(
            None, {self.input_name: np.zeros((1, 3, 256, 256), dtype=np.float32)}
        )

    def apply(self, face, matrix, alpha, strength):
        from modules.processors.frame._onnx_enhancer import (
            preprocess_face,
            postprocess_face,
        )

        enlarged = cv2.resize(face, (256, 256), interpolation=cv2.INTER_CUBIC)
        blob = preprocess_face(enlarged, 256)
        result = self.session.run(None, {self.input_name: blob})[0]
        restored = postprocess_face(result)
        restored = cv2.addWeighted(restored, strength, enlarged, 1 - strength, 0)
        # The original transform maps the frame into a 128px swap crop. Scale
        # both linear and translation terms when returning a 256px restored crop.
        scale = 256 / face.shape[0]
        return restored, matrix * scale, cv2.resize(alpha, (256, 256))
