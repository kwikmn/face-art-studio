from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import cv2
from modules.live_face import LiveFaceProcessor, CudaReplaySession
from modules.studio.live_enhancement import LiveGPEN


class LiveEnhancementTests(unittest.TestCase):
    def test_replay_packs_transposed_inputs_on_every_frame(self):
        # Emulate the raw memory copy made by OrtValue.update_inplace. Distinct
        # channel/pixel values expose corruption that constant test crops hide.
        import ctypes

        class Buffer:
            def update_inplace(self, value):
                raw = ctypes.string_at(value.ctypes.data, value.nbytes)
                self.value = np.frombuffer(raw, dtype=value.dtype).reshape(value.shape).copy()

        buffer = Buffer()
        replay = CudaReplaySession.__new__(CudaReplaySession)
        replay.failed = False
        replay.inputs = {"input": buffer}
        replay.bindings = SimpleNamespace(copy_outputs_to_cpu=lambda: [buffer.value])
        replay.session = SimpleNamespace(run_with_iobinding=lambda *_: None)
        for offset in (0, 100):
            hwc = np.arange(48, dtype=np.float32).reshape(4, 4, 3) + offset
            blob = hwc.transpose(2, 0, 1)[None]
            self.assertFalse(blob.flags.c_contiguous)
            result = replay.run(None, {"input": blob})[0]
            np.testing.assert_array_equal(result, blob)

    def test_zero_strength_preserves_swapped_crop_and_scales_transform(self):
        enhancer = LiveGPEN.__new__(LiveGPEN)
        enhancer.input_name = "input"
        enhancer.session = SimpleNamespace(
            run=lambda *args: [np.zeros((1, 3, 256, 256), np.float32)]
        )
        face = np.full((128, 128, 3), 70, np.uint8)
        matrix = np.array([[1.0, 0.0, -10.0], [0.0, 1.0, -20.0]])
        alpha = np.full((128, 128), 255, np.uint8)
        result, transform, mask = enhancer.apply(face, matrix, alpha, 0)
        np.testing.assert_array_equal(result, cv2.resize(face, (256, 256)))
        np.testing.assert_array_equal(transform, matrix * 2)
        self.assertEqual(mask.shape, (256, 256))
        np.testing.assert_array_equal(
            matrix, np.array([[1.0, 0.0, -10.0], [0.0, 1.0, -20.0]])
        )

    def test_enhancer_failure_keeps_swapped_face_not_original(self):
        processor = LiveFaceProcessor("unused", width=100, height=100)
        processor.source = object()
        processor.detector = SimpleNamespace(
            detect=lambda *a, **k: (
                np.array([[20, 20, 80, 80, 0.9]]),
                np.array([[[35, 35], [65, 35], [50, 50], [40, 65], [60, 65]]]),
            )
        )
        processor.swapper = SimpleNamespace(
            get=lambda *a, **k: (
                np.full((128, 128, 3), 70, np.uint8),
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            )
        )
        processor.alpha = np.full((128, 128), 255, np.uint8)

        class Broken:
            def apply(self, *args):
                raise RuntimeError("enhancer failure")

        processor.set_live_enhancement(Broken())
        with patch.dict(
            "sys.modules",
            {
                "insightface.app.common": SimpleNamespace(
                    Face=lambda **k: SimpleNamespace(**k)
                )
            },
        ):
            result = processor.process(np.full((100, 100, 3), 199, np.uint8))
        self.assertEqual(processor.enhancement_error, "enhancer failure")
        self.assertTrue(np.all(result.image == 70))


if __name__ == "__main__":
    unittest.main()
