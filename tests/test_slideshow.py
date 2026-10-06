import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from contextlib import nullcontext
import threading
import hashlib
import json
import numpy as np
from PIL import Image
from modules.studio.slideshow import Slideshow, ImageStore, decode_image
from modules.studio.background import BackgroundEffect, BackgroundSettings
from modules.live_face import LiveFaceProcessor


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Store:
    def __init__(self):
        self.values = {}
        self.requests = []

    def get(self, key):
        return self.values.get(key)

    def request(self, key, priority=False):
        self.requests.append(key)

    def close(self):
        pass


class SlideshowTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = Store()
        self.slide = Slideshow(
            dict(paths=["a", "b", "c"], interval=2, fade=1), self.store, self.clock
        )
        self.slide.set_size(80, 48)
        for i, path in enumerate(["a", "b", "c"]):
            self.store.values[(path, (80, 48))] = (
                np.full((48, 80, 3), i * 100, np.uint8),
                None,
            )
        self.slide.tick()

    def test_manual_crossfade_background_only(self):
        foreground = np.full((48, 80, 3), 77, np.uint8)
        mask = np.zeros((48, 80), np.float32)
        mask[:, :40] = 1
        effect = BackgroundEffect()
        effect.slideshow = self.slide
        settings = BackgroundSettings(
            mode="slideshow", softness=0, smoothing=0, chair_cleanup=0, tightness=0
        )
        self.slide.navigate(1)
        self.slide.tick()
        self.clock.now = 0.5
        with patch.object(effect, "_mask", return_value=mask):
            out = effect.apply(foreground, settings)
        np.testing.assert_array_equal(out[:, :40], foreground[:, :40])
        self.assertTrue(np.all(out[:, 40:] == 50))
        self.clock.now = 1
        self.slide.tick()
        self.assertIsNone(self.slide.old)

    def test_delayed_tick_drops_intervals(self):
        self.slide.play(True)
        self.clock.now = 100
        self.slide.tick()
        self.assertEqual(self.slide.active, "b")
        self.assertEqual(self.slide.deadline, 102)
        self.slide.tick()
        self.assertEqual(self.slide.active, "b")

    def test_pause_freezes_transition_and_resume_rebases(self):
        self.slide.play(True)
        self.slide.navigate(1)
        self.slide.tick()
        self.clock.now = 0.25
        self.slide.play(False)
        before = self.slide.frame(80, 48).copy()
        self.clock.now = 100
        np.testing.assert_array_equal(before, self.slide.frame(80, 48))
        self.slide.play(True)
        self.clock.now = 100.25
        self.assertTrue(np.all(self.slide.frame(80, 48) == 50))

    def test_interval_change_resets_from_now(self):
        self.slide.play(True)
        self.clock.now = 1
        self.slide.configure(5, 0, True)
        self.assertEqual(self.slide.deadline, 6)
        self.clock.now = 5
        self.slide.tick()
        self.assertEqual(self.slide.active, "a")
        self.clock.now = 6
        self.slide.tick()
        self.assertEqual(self.slide.active, "b")

    def test_reordering_preserves_active_removal_selects_neighbour(self):
        self.slide.select("b")
        self.slide.tick()
        self.slide.set_paths(["c", "b", "a"])
        self.slide.tick()
        self.assertEqual(self.slide.active, "b")
        self.slide.set_paths(["c", "a"])
        self.slide.tick()
        self.assertEqual(self.slide.active, "a")

    def test_rapid_navigation_latest_target_wins(self):
        self.slide.navigate(1)
        self.slide.navigate(1)
        self.slide.tick()
        self.assertEqual(self.slide.active, "c")

    def test_missing_slide_keeps_valid_then_advances(self):
        self.store.values[("b", (80, 48))] = (None, "missing")
        self.slide.play(True)
        self.clock.now = 2
        self.slide.tick()
        self.assertEqual(self.slide.active, "a")
        self.assertEqual(self.slide.error, "missing")
        self.clock.now = 4
        self.slide.tick()
        self.assertEqual(self.slide.active, "c")

    def test_pending_load_never_blocks_or_exposes_room(self):
        self.store.values.clear()
        self.slide.set_size(40, 24)
        self.slide.tick()
        effect = BackgroundEffect()
        effect.slideshow = self.slide
        frame = np.full((24, 40, 3), 199, np.uint8)
        with patch.object(effect, "_mask", return_value=np.zeros((24, 40), np.float32)):
            out = effect.apply(
                frame,
                BackgroundSettings(
                    mode="slideshow",
                    softness=0,
                    smoothing=0,
                    chair_cleanup=0,
                    tightness=0,
                ),
            )
        self.assertFalse(np.any(out == 199))
        self.assertIsNone(self.slide.frame(40, 24))
        self.assertTrue(self.store.requests)

    def test_mask_failure_still_propagates_protected_boundary(self):
        processor = LiveFaceProcessor("unused")
        processor.slideshow = self.slide
        processor.set_background(BackgroundSettings(mode="slideshow"))
        with patch.object(
            BackgroundEffect, "_mask", side_effect=RuntimeError("bad mask")
        ):
            with self.assertRaisesRegex(RuntimeError, "bad mask"):
                processor._apply_background(np.zeros((48, 80, 3), np.uint8))

    def test_empty_one_item_no_loop_and_restart(self):
        self.slide.set_paths([])
        self.slide.play(True)
        self.slide.tick()
        self.assertIsNone(self.slide.frame(80, 48))
        self.assertFalse(self.slide.playing)
        self.slide.set_paths(["a"])
        self.slide.configure(2, 0, False)
        self.slide.tick()
        self.slide.play(True)
        self.clock.now = 3
        self.slide.tick()
        self.assertFalse(self.slide.playing)
        restarted = Slideshow(self.slide.settings(), self.store, self.clock)
        restarted.set_size(80, 48)
        restarted.tick()
        self.assertEqual(restarted.active, "a")
        self.assertFalse(restarted.playing)

    def test_decode_limits_corruption_and_sources_untouched(self):
        # Retain generated fixtures; do not delete anything during this pass.
        with nullcontext(
            tempfile.mkdtemp(prefix="faceart-candidate4-tests-")
        ) as directory:
            root = Path(directory)
            good = root / "good.png"
            Image.new("RGB", (70, 90), "red").save(good)
            before = hashlib.sha256(good.read_bytes()).hexdigest()
            image = decode_image(good, (80, 48))
            self.assertEqual(image.shape, (48, 80, 3))
            self.assertEqual(before, hashlib.sha256(good.read_bytes()).hexdigest())
            bad = root / "bad.png"
            bad.write_bytes(b"broken")
            with self.assertRaises(Exception):
                decode_image(bad, (80, 48))
            with self.assertRaises(FileNotFoundError):
                decode_image(root / "missing.png", (80, 48))
            with patch("modules.studio.slideshow.Image.open") as opener:
                opener.return_value.__enter__.return_value.size = (5000, 5000)
                opener.return_value.__enter__.return_value.width = 5000
                opener.return_value.__enter__.return_value.height = 5000
                with self.assertRaisesRegex(ValueError, "megapixel"):
                    decode_image(good, (80, 48))
            with patch.object(Path, "stat") as stat:
                stat.return_value.st_size = 33 * 1024 * 1024
                with self.assertRaisesRegex(ValueError, "32 MB"):
                    decode_image(good, (80, 48))

    def test_loader_nonblocking_bounded_and_stale_latest(self):
        entered = threading.Event()
        release = threading.Event()

        def decoder(path, size):
            entered.set()
            release.wait(2)
            return np.zeros((size[1], size[0], 3), np.uint8)

        store = ImageStore(decoder)
        try:
            store.request(("a", (80, 48)))
            self.assertTrue(entered.wait(1))
            for i in range(50):
                store.request((str(i), (80, 48)), True)
            self.assertLessEqual(len(store.queue), 3)
            self.assertIsNone(store.get(("49", (80, 48))))
            release.set()
            store.close()
            store.thread.join(2)
            self.assertFalse(store.thread.is_alive())
            self.assertLessEqual(len(store.cache), 3)
        finally:
            release.set()
            store.close()

    def test_preload_next_and_cut_without_fade(self):
        self.assertIn(("b", (80, 48)), self.store.requests)
        self.slide.configure(2, 0, True)
        self.slide.navigate(1)
        self.slide.tick()
        self.assertTrue(np.all(self.slide.frame(80, 48) == 100))

    def test_corrupt_settings_and_no_loop_missing_last(self):
        corrupt = Slideshow(
            dict(paths=None, interval="bad", fade=float("nan")), self.store, self.clock
        )
        self.assertEqual(corrupt.paths, [])
        self.assertEqual(corrupt.interval, 10)
        self.slide.set_paths(["a", "b"])
        self.store.values[("b", (80, 48))] = (None, "missing")
        self.slide.configure(2, 0, False)
        self.slide.tick()
        self.slide.play(True)
        self.clock.now = 2
        self.slide.tick()
        self.clock.now = 4
        self.slide.tick()
        self.assertFalse(self.slide.playing)

    def test_new_processor_restart_and_resolution_change(self):
        self.slide.configure(2, 0, True)
        self.slide.select("b")
        self.slide.tick()
        cfg = BackgroundSettings(
            mode="slideshow", softness=0, smoothing=0, chair_cleanup=0, tightness=0
        )
        for _ in range(2):
            processor = LiveFaceProcessor("unused")
            processor.slideshow = self.slide
            processor.set_background(cfg)
            with patch.object(
                BackgroundEffect, "_mask", return_value=np.zeros((48, 80), np.float32)
            ):
                result = processor._apply_background(
                    np.full((48, 80, 3), 199, np.uint8)
                )
            self.assertTrue(np.all(result == 100))
            self.assertEqual(self.slide.active, "b")
        self.slide.set_size(40, 24)
        self.assertIsNone(self.slide.frame(40, 24))
        self.store.values[("b", (40, 24))] = (np.full((24, 40, 3), 100, np.uint8), None)
        self.slide.tick()
        self.assertEqual(self.slide.frame(40, 24).shape, (24, 40, 3))

    def test_pending_active_removal_save_restart_keeps_surviving_neighbour(self):
        self.slide.configure(2, 0, True)
        self.slide.select("b")
        self.slide.tick()
        self.store.values.pop(("c", (80, 48)))
        self.slide.set_paths(["a", "c"])
        self.slide.tick()
        self.assertEqual(self.slide.pending, "c")
        self.assertEqual(self.slide.active, "b")
        self.assertTrue(np.all(self.slide.frame(80, 48) == 100))
        saved = json.loads(json.dumps(self.slide.settings()))
        self.assertEqual(saved["active"], "c")
        restarted = Slideshow(saved, self.store, self.clock)
        restarted.set_size(80, 48)
        restarted.tick()
        self.assertEqual(restarted.pending, "c")
        self.assertEqual(restarted.active, "c")
        self.assertIsNone(restarted.frame(80, 48))
        self.assertFalse(restarted.playing)
        self.store.values[("c", (80, 48))] = (np.full((48, 80, 3), 200, np.uint8), None)
        restarted.tick()
        self.assertEqual(restarted.active, "c")
        self.assertTrue(np.all(restarted.frame(80, 48) == 200))

    def test_failed_removal_neighbour_save_restart_retries_survivor(self):
        self.slide.configure(2, 0, True)
        self.slide.select("b")
        self.slide.tick()
        self.store.values[("c", (80, 48))] = (None, "corrupt neighbour")
        self.slide.set_paths(["a", "c"])
        self.slide.tick()
        self.assertEqual(self.slide.active, "b")
        self.assertIsNone(self.slide.pending)
        self.assertTrue(np.all(self.slide.frame(80, 48) == 100))
        saved = json.loads(json.dumps(self.slide.settings()))
        self.assertEqual(saved["active"], "c")
        restarted = Slideshow(saved, self.store, self.clock)
        restarted.set_size(80, 48)
        restarted.tick()
        self.assertEqual(restarted.active, "c")
        self.assertEqual(restarted.error, "corrupt neighbour")
        self.assertIsNone(restarted.frame(80, 48))
        self.assertFalse(restarted.playing)


if __name__ == "__main__":
    unittest.main()
