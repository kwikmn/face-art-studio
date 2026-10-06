import unittest
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from modules.studio.background import BackgroundEffect, BackgroundSettings
from modules.live_face import LiveFaceProcessor


class BackgroundTests(unittest.TestCase):
    def test_off_returns_identical_frame_without_loading(self):
        effect = BackgroundEffect()
        frame = np.full((48, 80, 3), 77, np.uint8)
        with patch.object(effect, "_load", side_effect=AssertionError("must not load")):
            self.assertIs(effect.apply(frame, BackgroundSettings()), frame)
        processor = LiveFaceProcessor("unused")
        processor.set_background(BackgroundSettings())
        self.assertIs(processor._apply_background(frame), frame)
        self.assertIsNone(processor.background_effect)

    def test_color_uses_protected_foreground_and_rgb_color(self):
        effect = BackgroundEffect()
        frame = np.full((144, 256, 3), 77, np.uint8)
        mask = np.zeros((144, 256), np.float32)
        mask[:, :128] = 1
        with patch.object(effect, "_mask", return_value=mask):
            result = effect.apply(
                frame, BackgroundSettings(mode="color", color="#ff0000", softness=0)
            )
        np.testing.assert_array_equal(result[:, :128], frame[:, :128])
        self.assertTrue(np.all(result[:, 128:] == [0, 0, 255]))
        self.assertTrue(np.all(frame == 77))

    def test_image_covers_frame_and_is_cached(self):
        effect = BackgroundEffect()
        frame = np.full((144, 256, 3), 77, np.uint8)
        image = np.full((100, 100, 3), [10, 30, 50], np.uint8)
        with (
            patch.object(
                effect, "_mask", return_value=np.zeros((144, 256), np.float32)
            ),
            patch(
                "modules.studio.background.imread_unicode", return_value=image
            ) as read,
        ):
            settings = BackgroundSettings(mode="image", image="test.jpg", softness=0)
            result = effect.apply(frame, settings)
            effect.apply(frame, settings)
            self.assertEqual(read.call_count, 1)
        self.assertEqual(result.shape, frame.shape)
        self.assertTrue(np.all(result == [10, 30, 50]))

    def test_temporal_state_resets_on_configuration_and_off(self):
        effect = BackgroundEffect()
        frame = np.zeros((144, 256, 3), np.uint8)
        settings = BackgroundSettings(mode="color", softness=0, smoothing=50)
        with patch.object(
            effect,
            "_mask",
            side_effect=[
                np.ones((144, 256), np.float32),
                np.zeros((144, 256), np.float32),
                np.zeros((144, 256), np.float32),
            ],
        ):
            effect.apply(frame, settings)
            effect.apply(frame, settings)
            self.assertTrue(np.all(effect.previous == 0.5))
            effect.apply(frame, replace(settings, color="#ffffff"))
            self.assertFalse(np.any(effect.previous))
        effect.apply(frame, BackgroundSettings())
        self.assertIsNone(effect.previous)

    def test_blur_reduces_background_detail(self):
        effect = BackgroundEffect()
        checker = ((np.indices((144, 256)).sum(axis=0) % 2) * 255).astype(np.uint8)
        frame = np.repeat(checker[:, :, None], 3, axis=2)
        with patch.object(
            effect, "_mask", return_value=np.zeros((144, 256), np.float32)
        ):
            result = effect.apply(frame, BackgroundSettings(mode="blur"))
        self.assertLess(result.std(), frame.std() / 5)
        self.assertEqual(result.dtype, np.uint8)

    def test_face_guidance_suppresses_uncertain_chair_but_keeps_shoulders(self):
        effect = BackgroundEffect()
        probability = np.full((144, 256), 0.65, np.float32)
        settings = BackgroundSettings(
            mode="color", softness=0, chair_cleanup=70, tightness=25
        )
        box = (90, 35, 130, 85)
        baseline = effect._refine_mask(
            probability, replace(settings, chair_cleanup=0), (144, 256, 3), box
        )
        refined = effect._refine_mask(probability, settings, (144, 256, 3), box)
        self.assertLess(refined[55, 150], baseline[55, 150] / 2)
        self.assertAlmostEqual(refined[120, 150], baseline[120, 150])
        # A confident raised hand outside the shoulder guide is not hard-cut.
        probability[55, 180] = 1
        refined = effect._refine_mask(probability, settings, (144, 256, 3), box)
        self.assertEqual(refined[55, 180], 1)

    def test_actual_swap_mask_protects_face_core(self):
        effect = BackgroundEffect()
        probability = np.zeros((144, 256), np.float32)
        matrix = np.array([[1, 0, -94], [0, 1, -45]], np.float32)
        alpha = np.full((32, 32), 255, np.uint8)
        result = effect._refine_mask(
            probability,
            BackgroundSettings(mode="color"),
            (144, 256, 3),
            (90, 35, 130, 85),
            (matrix, alpha),
        )
        self.assertEqual(result[55, 105], 1)
        self.assertEqual(result[55, 150], 0)
        np.testing.assert_array_equal(alpha, np.full((32, 32), 255, np.uint8))

    def test_zero_refinement_matches_previous_mask_math(self):
        effect = BackgroundEffect()
        probability = np.linspace(0, 1, 144 * 256, dtype=np.float32).reshape(144, 256)
        settings = BackgroundSettings(
            mode="color", softness=0, chair_cleanup=0, tightness=0
        )
        result = effect._refine_mask(
            probability, settings, (144, 256, 3), (90, 35, 130, 85)
        )
        expected = np.clip((probability - 0.25) / 0.5, 0, 1)
        expected = expected * expected * (3 - 2 * expected)
        np.testing.assert_array_equal(result, expected)

    def test_model_gets_rgb_normalized_and_rejects_invalid_output(self):
        effect = BackgroundEffect()
        received = {}

        def run(_, inputs):
            received.update(inputs)
            return [np.full((1, 144, 256, 2), np.nan, np.float32)]

        effect.session = SimpleNamespace(run=run)
        effect.input_name = "input"
        with self.assertRaisesRegex(RuntimeError, "Invalid background"):
            effect._mask(np.full((144, 256, 3), [0, 0, 255], np.uint8))
        np.testing.assert_array_equal(received["input"][0, 0, 0], [1, 0, 0])

    def test_missing_image_and_model_fail_without_returning_original(self):
        processor = LiveFaceProcessor("unused")
        processor.set_background(BackgroundSettings(mode="color"))
        processor.background_effect = SimpleNamespace(
            apply=lambda *_: (_ for _ in ()).throw(RuntimeError("Model failed"))
        )
        with self.assertRaisesRegex(RuntimeError, "Model failed"):
            processor._apply_background(np.zeros((48, 80, 3), np.uint8))
        self.assertEqual(processor.background_error, "Model failed")
        effect = BackgroundEffect()
        with patch("modules.studio.background.imread_unicode", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "available background image"):
                effect._image("missing.jpg", 100, 100)


if __name__ == "__main__":
    unittest.main()
