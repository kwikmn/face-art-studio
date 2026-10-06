import tempfile
import unittest
import json
from pathlib import Path
import numpy as np
from PIL import Image
from modules.studio.reference_camera_worker import square_crop
from modules.studio.snapshot_blend import align_edit, blend_pair, save_blend, file_hash


class SnapshotBlendTests(unittest.TestCase):
    def test_square_snapshot_keeps_center_pixels_without_resizing(self):
        frame = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
        result = square_crop(frame)
        np.testing.assert_array_equal(result, frame[:, 2:8])
        self.assertFalse(np.shares_memory(result, frame))

    def test_alignment_reverses_known_translation(self):
        original = np.zeros((120, 120, 3), np.uint8)
        changed = np.zeros_like(original)
        changed[65, 70] = 255
        points = np.array(
            [[35, 40], [75, 40], [55, 60], [40, 80], [70, 80]], np.float32
        )
        aligned, support, matrix = align_edit(
            original, changed, points, points + [10, 5]
        )
        np.testing.assert_allclose(matrix, [[1, 0, -10], [0, 1, -5]], atol=1e-5)
        np.testing.assert_array_equal(aligned[60, 60], [255] * 3)
        self.assertEqual(support[60, 60], 1)
        self.assertEqual(support[-1, -1], 0)

    def test_blend_keeps_original_outside_mask_and_full_edit_in_center(self):
        original = np.full((160, 160, 3), 31, np.uint8)
        edited = np.full_like(original, 219)
        pair = dict(
            original=original,
            aligned=edited,
            support=np.ones((160, 160)),
            box=[40, 30, 120, 130],
        )
        result, mask = blend_pair(pair)
        np.testing.assert_array_equal(result[mask == 0], original[mask == 0])
        np.testing.assert_array_equal(result[mask == 1], edited[mask == 1])
        self.assertGreater(np.count_nonzero((mask > 0) & (mask < 1)), 200)
        np.testing.assert_array_equal(original, np.full_like(original, 31))
        with self.assertRaises(ValueError):
            blend_pair(pair, coverage=0)

    def test_saved_blend_has_lineage_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.png"
            edit = Path(tmp) / "edit.png"
            Image.new("RGB", (32, 32), "red").save(base)
            Image.new("RGB", (32, 32), "blue").save(edit)
            hashes = [file_hash(base), file_hash(edit)]
            output = save_blend(
                np.full((32, 32, 3), 70, np.uint8),
                Path(tmp) / "out",
                {},
                base,
                edit,
                dict(coverage=1, feather=0.25, offset_x=0, offset_y=0),
                np.eye(2, 3),
            )
            with Image.open(output) as image:
                data = json.loads(image.info["FaceArt Studio"])
                self.assertEqual(image.size, (32, 32))
            self.assertEqual(data["mode"], "blend")
            self.assertEqual(data["reference_sha256"], hashes[0])
            self.assertEqual(data["edited_sha256"], hashes[1])
            self.assertEqual([file_hash(base), file_hash(edit)], hashes)
            self.assertFalse(list((Path(tmp) / "out").glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
