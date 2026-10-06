from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import setup_windows


class SetupPlatformTests(unittest.TestCase):
    def test_insufficient_space_stops_before_installation(self):
        with tempfile.TemporaryDirectory(prefix='fas-test-') as folder:
            root = Path(folder)
            with patch.object(setup_windows, 'ROOT', root), \
                 patch.object(setup_windows, 'VENV', root / 'venv'), \
                 patch.object(setup_windows, 'PYTHON', root / 'venv/Scripts/python.exe'), \
                 patch.object(setup_windows, 'supported_windows', return_value=True), \
                 patch.object(setup_windows, 'supported_python', return_value=True), \
                 patch.object(setup_windows, 'nvidia_gpu', return_value='GPU, 581.57'), \
                 patch.object(setup_windows, 'cpp_tools', return_value=True), \
                 patch.object(setup_windows, 'obs_registered', return_value=False), \
                 patch.object(setup_windows.ctypes, 'WinDLL', create=True), \
                 patch.object(setup_windows.shutil, 'disk_usage', return_value=SimpleNamespace(free=11 * 1024**3)), \
                 patch.object(setup_windows, 'install_plan') as plan, \
                 patch.object(setup_windows.os, 'chdir'), \
                 patch.object(setup_windows.sys, 'argv', ['setup_windows.py']), \
                 patch.dict(os.environ, {'PIP_CACHE_DIR': str(root / 'cache')}):
                self.assertEqual(setup_windows.main(), 1)
                plan.assert_not_called()
                self.assertFalse((root / 'venv').exists())
                self.assertFalse((root / 'state').exists())

    def test_supported_windows_workstations_and_server2022(self):
        cases = [(10, 19045, 1, True), (10, 22631, 1, True),
                 (10, 20348, 3, True), (10, 20348, 2, True),
                 (10, 17763, 3, False), (10, 26100, 3, False),
                 (6, 9600, 1, False)]
        with patch.object(setup_windows.sys, 'platform', 'win32'):
            for major, build, product_type, expected in cases:
                with self.subTest(major=major, build=build, product_type=product_type):
                    version = SimpleNamespace(major=major, build=build, product_type=product_type)
                    self.assertEqual(setup_windows.supported_windows(version), expected)
        with patch.object(setup_windows.sys, 'platform', 'linux'):
            self.assertFalse(setup_windows.supported_windows())

    def test_driver_minimum_and_unreadable_output(self):
        cases = {'GPU, 570.65': True, 'GPU, 581.57': True,
                 'GPU, 570.64': False, 'GPU, 569.99': False,
                 'GPU A, 581.57\nGPU B, 581.57': True,
                 'GPU A, 581.57\nGPU B, 527.41': False,
                 '': False, 'GPU, unknown': False, 'GPU': False,
                 'GPU, 570': False, 'GPU, 570.65.1': False}
        for output, expected in cases.items():
            with self.subTest(output=output):
                self.assertEqual(setup_windows.supported_driver(output), expected)


if __name__ == '__main__':
    unittest.main()
import os
from pathlib import Path
import tempfile
