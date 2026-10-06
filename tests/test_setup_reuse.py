import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import setup_windows


class SetupReuseTests(unittest.TestCase):
    def test_absent_environment_does_not_probe_or_skip_compiler(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(setup_windows, 'PYTHON', Path(folder) / 'missing.exe'), patch.object(setup_windows.subprocess, 'run') as run:
                self.assertFalse(setup_windows.existing_insightface())
                run.assert_not_called()

    def test_compatible_compiled_package_can_be_reused_without_install(self):
        with tempfile.TemporaryDirectory(prefix='FaceArt setup ') as folder:
            python = Path(folder) / 'python.exe'
            python.touch()
            with patch.object(setup_windows, 'PYTHON', python), patch.object(setup_windows.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
                self.assertTrue(setup_windows.existing_insightface())
                arguments = run.call_args.args[0]
                self.assertEqual(arguments[:3], [str(python), '-s', '-c'])
                self.assertIn("d.version=='0.7.3'", arguments[3])
                self.assertIn('mesh_core_cython', arguments[3])
                self.assertIn('is_file()', arguments[3])

    def test_missing_or_broken_package_still_requires_compiler(self):
        with tempfile.TemporaryDirectory() as folder:
            python = Path(folder) / 'python.exe'
            python.touch()
            with patch.object(setup_windows, 'PYTHON', python), patch.object(setup_windows.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
                self.assertFalse(setup_windows.existing_insightface())
            with patch.object(setup_windows, 'PYTHON', python), patch.object(setup_windows.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 15)):
                self.assertFalse(setup_windows.existing_insightface())


if __name__ == '__main__':
    unittest.main()
