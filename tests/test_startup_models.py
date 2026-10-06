import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from scripts import check_install, launch_studio


class StartupModelTests(unittest.TestCase):
    def test_damaged_optional_model_does_not_block_baseline_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'models').mkdir()
            (root / 'models/required.onnx').write_bytes(b'good')
            (root / 'models/optional.onnx').write_bytes(b'bad')
            (root / 'models/downloads.json').write_text(json.dumps({'models': [
                {'path': 'models/required.onnx', 'required': True, 'bytes': 4, 'url': 'https://example.test/required'},
                {'path': 'models/optional.onnx', 'required': False, 'bytes': 8, 'url': 'https://example.test/optional'},
            ]}))
            with patch.object(check_install, 'ROOT', root), patch.object(launch_studio, 'ROOT', root), patch.object(launch_studio.subprocess, 'call', return_value=0) as launch:
                self.assertEqual(check_install.model_problems(required_only=True), [])
                self.assertIn('Wrong size: models/optional.onnx', check_install.model_problems()[0])
                self.assertEqual(launch_studio.main(), 0)
                launch.assert_called_once()

    def test_missing_required_model_still_prevents_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'models').mkdir()
            (root / 'models/downloads.json').write_text(json.dumps({'models': [
                {'path': 'models/required.onnx', 'required': True, 'bytes': 4, 'url': 'https://example.test/required'},
            ]}))
            with patch.object(check_install, 'ROOT', root), patch.object(launch_studio, 'ROOT', root), patch.object(launch_studio.subprocess, 'call') as launch:
                self.assertEqual(launch_studio.main(), 2)
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
