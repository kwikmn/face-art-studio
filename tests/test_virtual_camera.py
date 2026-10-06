import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.virtual_camera import VirtualCameraOutput


class VirtualCameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = Mock(width=8, height=6, device="OBS Virtual Camera")
        self.backend = types.SimpleNamespace(
            Camera=Mock(return_value=self.camera),
            PixelFormat=types.SimpleNamespace(BGR="BGR"),
        )
        self.patch = patch.dict(sys.modules, {"pyvirtualcam": self.backend})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.ownership = patch("modules.virtual_camera.check_obs_available")
        self.ownership.start()
        self.addCleanup(self.ownership.stop)
        self.output = VirtualCameraOutput()
        self.addCleanup(self.output.close)

    def test_bgr_frames_and_resize_keep_camera_dimensions(self):
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        self.output.send(frame)
        self.output.send(np.zeros((12, 16, 3), dtype=np.uint8))
        self.backend.Camera.assert_called_once_with(
            width=8,
            height=6,
            fps=30,
            fmt="BGR",
            backend="obs",
            device="OBS Virtual Camera",
        )
        np.testing.assert_array_equal(self.camera.send.call_args_list[0].args[0], frame)
        self.assertEqual(self.camera.send.call_args.args[0].shape, (6, 8, 3))
        self.camera.sleep_until_next_frame.assert_not_called()

    def test_close_is_idempotent_and_output_can_restart(self):
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        self.output.send(frame)
        self.output.close()
        self.output.close()
        self.camera.close.assert_called_once()
        self.output.send(frame)
        self.assertEqual(self.backend.Camera.call_count, 2)

    def test_busy_precheck_never_constructs_output(self):
        with patch(
            "modules.virtual_camera.check_obs_available",
            side_effect=RuntimeError("busy"),
        ):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.output.send(np.zeros((6, 8, 3), np.uint8))
        self.backend.Camera.assert_not_called()
        self.assertFalse(self.output.info()["running"])

    def test_startup_failure_stays_off_and_info_reports_actual_device(self):
        self.backend.Camera.side_effect = RuntimeError("driver unavailable")
        with self.assertRaisesRegex(RuntimeError, "driver unavailable"):
            self.output.send(np.zeros((6, 8, 3), np.uint8))
        self.assertFalse(self.output.info()["running"])
        self.output.close()
        self.backend.Camera.side_effect = None
        self.output.send(np.zeros((6, 8, 3), np.uint8))
        self.assertEqual(self.output.info()["device"], "OBS Virtual Camera")
        self.assertEqual(self.output.info()["frames_sent"], 1)
        self.output.close()
        self.assertFalse(self.output.info()["running"])

    def test_backend_failure_can_be_closed_and_retried(self):
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        self.camera.send.side_effect = RuntimeError("camera unavailable")
        with self.assertRaises(RuntimeError):
            self.output.send(frame)
        self.output.close()
        self.camera.close.assert_called_once()
        self.camera.send.side_effect = None
        self.output.send(frame)
        self.assertEqual(self.backend.Camera.call_count, 2)


if __name__ == "__main__":
    unittest.main()
