import json
import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from modules.studio.facegen import Generator, validate_request, MODEL_ID, REVISION


class FaceGenerationTests(unittest.TestCase):
    def test_invalid_requests_rejected(self):
        for fields in (
            {"prompt": ""},
            {"size": 513},
            {"count": 0},
            {"memory": "unknown"},
            {"seed": -1},
            {"seed": 2**32},
            {"reference": "missing-photo.png"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_request(
                    dict({"prompt": "A fictional adult portrait"}, **fields)
                )

    def test_random_seed_is_resolved_once_for_a_batch(self):
        with patch("modules.studio.facegen.secrets.randbelow", return_value=123):
            self.assertEqual(validate_request({"prompt": "portrait"})["seed"], 123)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'Optional Face Lab needs PyTorch')
    def test_editing_metadata_and_batch_seed_wrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / "reference.png"
            Image.new("RGB", (1600, 800), "red").save(reference)
            engine = Generator()
            engine.resolved = "model"
            calls = []
            events = []

            def pipeline(**kwargs):
                calls.append(kwargs)
                kwargs["callback_on_step_end"](None, 3, None, {})
                return SimpleNamespace(images=[Image.new("RGB", (512, 512), "green")])

            engine.pipe = pipeline

            class Seed:
                def __init__(self, device):
                    self.device = device

                def manual_seed(self, value):
                    self.seed = value
                    return self

            with (
                patch.object(engine, "load"),
                patch("torch.Generator", Seed),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.cuda.max_memory_allocated", return_value=1024**3),
            ):
                engine.generate(
                    dict(
                        prompt="Make the eyebrows bushier",
                        reference=str(reference),
                        size=512,
                        count=2,
                        seed=2**32 - 1,
                        memory="model",
                    ),
                    Path(tmp) / "results",
                    lambda kind, **data: events.append((kind, data)),
                )
            results = [data for kind, data in events if kind == "result"]
            self.assertEqual(len(results), 2)
            self.assertEqual([call["generator"].seed for call in calls], [2**32 - 1, 0])
            self.assertEqual(calls[0]["image"].size, (1024, 512))
            self.assertEqual(calls[0]["num_inference_steps"], 4)
            self.assertEqual(calls[0]["guidance_scale"], 1)
            with Image.open(results[0]["path"]) as png:
                metadata = json.loads(png.info["FaceArt Studio"])
            self.assertEqual(metadata["model"], MODEL_ID)
            self.assertEqual(metadata["revision"], REVISION)
            self.assertEqual(metadata["mode"], "img2img")
            self.assertEqual(len(metadata["reference_sha256"]), 64)
            self.assertEqual(events[-1][0], "done")
            self.assertFalse(list((Path(tmp) / "results").glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
