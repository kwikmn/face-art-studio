import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from modules.live_pipeline import LiveConfig, LivePipeline, ProcessedFrame
from modules.live_face import LiveFaceProcessor, validate_face, paste_checked


class FakeCapture:
    def __init__(self):
        self.fail = False
        self.released = False

    def start(self, width, height, fps):
        return True

    def read(self):
        time.sleep(0.008)
        return not self.fail, np.full((60, 80, 3), 199, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeProcessor:
    def __init__(self):
        self.mode = 'valid'
        self.delay = 0
        self.setup_error = False

    def prepare(self):
        if self.setup_error:
            raise RuntimeError('missing model')

    def process(self, frame):
        time.sleep(self.delay)
        if self.mode == 'error':
            raise RuntimeError('inference error')
        if self.mode == 'blocked':
            return ProcessedFrame(None, 'No face')
        if self.mode == 'invalid':
            return frame
        return ProcessedFrame(np.full_like(frame, getattr(self, "color", 88)))

    def prepare_source_image(self, path):
        time.sleep(.15)
        if path == 'bad':
            raise ValueError('Invalid image')
        return int(path)

    def activate_source(self, face, path):
        self.color = face
        self.activation_thread = threading.current_thread().name


class FakeOutput:
    def __init__(self):
        self.frames = []
        self.fail = False
        self.closed = 0

    def send(self, frame):
        if self.fail:
            raise RuntimeError('backend unavailable')
        self.frames.append((time.perf_counter(), frame.copy(), threading.current_thread().name))

    def close(self):
        self.closed += 1


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('Timed out waiting for condition')


class LivePipelineTests(unittest.TestCase):
    def setUp(self):
        self.cap, self.proc, self.out = FakeCapture(), FakeProcessor(), FakeOutput()
        self.session = LivePipeline(self.cap, self.proc, self.out, LiveConfig(width=80, height=60, fps=30, max_age=0.12, hold_last_good=0.12))
        self.session.enable_output(True)  # Explicit consent for this fake test output.
        self.addCleanup(self.session.stop)

    def assert_no_raw(self):
        for _, frame, _ in self.out.frames:
            self.assertFalse(np.all(frame == 199))

    def test_new_pipeline_output_defaults_off_until_explicit_start(self):
        output=FakeOutput();session=LivePipeline(FakeCapture(),FakeProcessor(),output,LiveConfig(width=80,height=60))
        self.addCleanup(session.stop)
        self.assertFalse(session.metrics()['output_enabled']);session.start()
        wait_until(lambda:session.metrics().get('safe_frames',0)>2)
        self.assertEqual(output.frames,[])
        session.enable_output(True);wait_until(lambda:len(output.frames)>2)
        self.assertTrue(session.metrics()['output_enabled']);session.enable_output(False)
        before=len(output.frames);time.sleep(.08);self.assertEqual(len(output.frames),before)
        for _,frame,_ in output.frames:
            self.assertFalse(np.all(frame==199))

    def test_startup_slate_and_ui_independent_pacing(self):
        self.proc.delay = 0.06
        self.session.start()
        wait_until(lambda: len(self.out.frames) > 10)
        np.testing.assert_array_equal(self.out.frames[0][1], self.session.slate)
        self.assertTrue(all(t == 'protected-output' for _, _, t in self.out.frames))
        self.assertTrue(any(np.all(f == 88) for _, f, _ in self.out.frames))
        times = [t for t, _, _ in self.out.frames]
        self.assertGreaterEqual(min(np.diff(times)), 0.025)
        self.assertGreater(self.session.metrics().get('dropped_capture', 0), 0)
        self.assert_no_raw()

    def test_failures_never_publish_original(self):
        for mode in ('blocked', 'error', 'invalid'):
            self.proc.mode = mode
            if not self.session._threads:
                self.session.start()
            before = len(self.out.frames)
            wait_until(lambda: len(self.out.frames) > before + 3)
            self.assertTrue(all(np.array_equal(f, self.session.slate) for _, f, _ in self.out.frames[before:]))
        self.assert_no_raw()

    def test_pause_invalidates_inflight_and_resume_requires_fresh_result(self):
        self.session.config = LiveConfig(width=80, height=60, max_age=.12, hold_last_good=5)
        self.proc.delay = 0.08
        self.session.start()
        wait_until(lambda: self.session.metrics().get('safe_frames', 0) > 0)
        self.session.pause(True)
        before = len(self.out.frames)
        time.sleep(0.18)
        self.assertTrue(all(np.array_equal(f, self.session.slate) for _, f, _ in self.out.frames[before:]))
        self.session.pause(False)
        wait_until(lambda: np.all(self.out.frames[-1][1] == 88))
        self.assert_no_raw()

    def test_stall_watchdog_and_disconnect(self):
        self.session.start()
        wait_until(lambda: self.session.metrics().get('safe_frames', 0) > 2)
        self.proc.delay = 0.4
        time.sleep(0.22)
        np.testing.assert_array_equal(self.out.frames[-1][1], self.session.slate)
        self.cap.fail = True
        wait_until(lambda: self.cap.released)
        time.sleep(0.3)
        np.testing.assert_array_equal(self.out.frames[-1][1], self.session.slate)
        self.assert_no_raw()

    def test_short_rejections_hold_only_last_swap_then_expire(self):
        self.session.config = LiveConfig(width=80, height=60, max_age=.12, hold_last_good=.4)
        self.session.start()
        wait_until(lambda: np.all(self.out.frames[-1][1] == 88) if self.out.frames else False)
        self.proc.mode = 'error'
        wait_until(lambda: self.session.metrics().get('blocked', 0) >= 1)
        time.sleep(.15)
        np.testing.assert_array_equal(self.out.frames[-1][1], np.full((60,80,3),88))
        self.assertGreater(self.session.metrics().get('held_frames', 0), 0)
        wait_until(lambda: np.array_equal(self.out.frames[-1][1], self.session.slate))
        self.assert_no_raw()

    def test_disconnect_bypasses_hold_and_invalidates_inflight(self):
        self.session.config = LiveConfig(width=80, height=60, hold_last_good=5)
        self.session.start()
        wait_until(lambda: self.session.metrics().get('safe_frames', 0) > 2)
        self.cap.fail = True
        wait_until(lambda: self.cap.released)
        time.sleep(.08)
        np.testing.assert_array_equal(self.out.frames[-1][1], self.session.slate)
        self.assert_no_raw()

    def test_live_source_load_keeps_publishing_and_activates_between_frames(self):
        self.session.start()
        wait_until(lambda: self.session.metrics().get('safe_frames', 0) > 2)
        before = len(self.out.frames)
        self.session.change_source('55')
        wait_until(lambda: self.session.metrics()['source_status'] == 'Face changed')
        wait_until(lambda: np.all(self.out.frames[-1][1] == 55))
        self.assertGreater(len(self.out.frames), before+2)
        self.assertTrue(any(np.all(f == 88) for _, f, _ in self.out.frames[before:]))
        self.assertEqual(self.proc.activation_thread, 'protected-processing')
        self.assert_no_raw()

    def test_invalid_live_source_retains_current_face(self):
        self.session.start()
        wait_until(lambda: self.session.metrics().get('safe_frames', 0) > 2)
        self.session.change_source('bad')
        wait_until(lambda: 'failed' in self.session.metrics()['source_status'])
        wait_until(lambda: np.all(self.out.frames[-1][1] == 88))
        self.assert_no_raw()

    def test_latest_source_selection_wins(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def load(path):
            if path == '55':
                entered.set()
                release.wait(2)
            return int(path)
        self.proc.prepare_source_image = load
        self.session.start()
        wait_until(lambda: self.session.metrics().get('safe_frames', 0) > 2)
        self.session.change_source('55')
        self.assertTrue(entered.wait(1))
        self.session.change_source('66')
        release.set()
        wait_until(lambda: np.all(self.out.frames[-1][1] == 66))
        self.assertFalse(any(np.all(f == 55) for _, f, _ in self.out.frames))
        self.assert_no_raw()

    def test_setup_failure_and_final_slate(self):
        self.proc.setup_error = True
        self.session.start()
        wait_until(lambda: len(self.out.frames) > 3)
        self.assertIn('missing model', self.session.metrics()['status'])
        self.assertTrue(self.session.stop())
        np.testing.assert_array_equal(self.out.frames[-1][1], self.session.slate)

    def test_backend_failure_and_explicit_retry(self):
        self.out.fail = True
        self.session.start()
        wait_until(lambda: self.session.metrics()['output_error'])
        self.assertFalse(self.session.metrics()['output_enabled'])
        self.out.fail = False
        self.session.enable_output(True)
        wait_until(lambda: len(self.out.frames) > 2)
        self.assertIsNone(self.session.metrics()['output_error'])
        self.assert_no_raw()


class ProtectedGeometryTests(unittest.TestCase):
    def test_zero_or_multiple_faces_are_blocked_before_swap(self):
        processor = LiveFaceProcessor('unused')
        processor.width, processor.height = 100, 100
        def face(**kwargs):
            return SimpleNamespace(**kwargs)
        for count in (0, 2):
            processor.detector = SimpleNamespace(detect=lambda *a, **k: (
                np.zeros((count, 5)), np.zeros((count, 5, 2))))
            with patch.dict('sys.modules', {'insightface.app.common': SimpleNamespace(Face=face)}):
                result = processor.process(np.zeros((100,100,3), dtype=np.uint8))
            self.assertIsNone(result.image)
            self.assertIn('face', result.reason.lower())

    def test_rejects_low_confidence_partial_face_and_bad_landmarks(self):
        face = SimpleNamespace(bbox=np.array([20, 20, 80, 80]),
                               kps=np.array([[35,35],[65,35],[50,50],[40,65],[60,65]]), det_score=.9)
        validate_face(face, (100,100,3))
        face.det_score = .4
        with self.assertRaises(ValueError):
            validate_face(face, (100,100,3))
        face.det_score = .9
        face.bbox[0] = -15
        with self.assertRaises(ValueError):
            validate_face(face, (100,100,3))

    def test_small_detector_envelope_overrun_requires_visible_landmarks(self):
        face = SimpleNamespace(bbox=np.array([20., -3., 80., 80.]),
                               kps=np.array([[35.,25.],[65.,25.],[50.,45.],[40.,65.],[60.,65.]]), det_score=.9)
        validate_face(face, (100,100,3))
        face.kps[0,1] = -1
        with self.assertRaises(ValueError):
            validate_face(face, (100,100,3))
        face.kps[0,1] = 25
        face.bbox[1] = -15
        with self.assertRaises(ValueError):
            validate_face(face, (100,100,3))

    def test_paste_rejects_empty_output_singular_transform_and_offscreen(self):
        frame = np.zeros((100,100,3), dtype=np.uint8)
        fake = np.full((32,32,3), 99, dtype=np.uint8)
        alpha = np.full((32,32), 255, dtype=np.uint8)
        matrix = np.array([[1.,0,0],[0,1.,0]])
        for bad_fake, bad_matrix in ((None,matrix), (fake,np.zeros((2,3))),
                                     (fake,np.array([[1.,0,-1000],[0,1.,-1000]]))):
            with self.assertRaises(ValueError):
                paste_checked(frame, bad_fake, bad_matrix, alpha)
        result = paste_checked(frame, fake, matrix, alpha)
        self.assertTrue(np.any(result != frame))
        self.assertTrue(np.all(frame == 0))


if __name__ == '__main__':
    unittest.main()
