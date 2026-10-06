import unittest
import cv2
import numpy as np
from modules.face_blending import face_mask
from modules.live_face import LiveFaceProcessor, paste_checked


class FaceMaskTests(unittest.TestCase):
    def test_zero_crop_edges_and_opaque_facial_core_at_all_strengths(self):
        for size in (128, 256):
            y, x = np.mgrid[:size, :size] / (size - 1)
            core = (x >= 0.34) & (x <= 0.66) & (y >= 0.34) & (y <= 0.75)
            for strength in (0, 0.6, 1):
                with self.subTest(size=size, strength=strength):
                    mask = face_mask(size, strength)
                    self.assertEqual(mask.dtype, np.uint8)
                    self.assertFalse(np.any(mask[[0, -1], :]))
                    self.assertFalse(np.any(mask[:, [0, -1]]))
                    self.assertTrue(np.all(mask[core] == 255))
                    enlarged = cv2.resize(mask, (size * 2, size * 2))
                    self.assertFalse(np.any(enlarged[[0, -1], :]))
                    self.assertFalse(np.any(enlarged[:, [0, -1]]))

    def test_jaw_adjustment_only_softens_lower_region(self):
        narrow, broad = face_mask(128, 0), face_mask(128, 1)
        np.testing.assert_array_equal(narrow[:57], broad[:57])
        self.assertTrue(np.all(broad <= narrow))
        self.assertGreater(np.count_nonzero(broad[80:] < narrow[80:]), 300)

    def test_warped_mask_removes_abrupt_crop_boundary(self):
        old = np.zeros((128, 128), np.uint8)
        cv2.ellipse(old, (64, 64), (56, 56), 0, 0, 360, 255, -1)
        old = cv2.GaussianBlur(old, (31, 31), 12)
        frame = np.zeros((400, 400, 3), np.uint8)
        crop = np.full((128, 128, 3), 255, np.uint8)
        matrix = np.array([[0.5, 0, -30], [0, 0.5, -30]], np.float32)
        before = paste_checked(frame, crop, matrix, old)
        after = paste_checked(frame, crop, matrix, face_mask(128))
        old_jump = np.abs(np.diff(before[188, :, 0].astype(int))).max()
        new_jump = np.abs(np.diff(after[188, :, 0].astype(int))).max()
        self.assertLess(new_jump, old_jump / 2)
        self.assertTrue(np.all(after[:60] == 0))

    def test_live_updates_replace_cached_mask_between_frames(self):
        processor = LiveFaceProcessor("unused")
        first = processor._mask_for(128)
        self.assertIs(first, processor._mask_for(128))
        processor.set_jaw_feathering(1)
        self.assertIs(first, processor.alpha)
        second = processor._mask_for(128)
        self.assertIsNot(first, second)
        np.testing.assert_array_equal(first, face_mask(128, 0.6))
        np.testing.assert_array_equal(second, face_mask(128, 1))
        with self.assertRaises(ValueError):
            processor.set_jaw_feathering(float("nan"))


if __name__ == "__main__":
    unittest.main()
