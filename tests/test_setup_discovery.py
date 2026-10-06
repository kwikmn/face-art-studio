import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == 'win32', 'Windows CMD launcher regression')
class SetupDiscoveryTests(unittest.TestCase):
    def run_launcher(self, candidate_exit):
        standard = Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Python/Python310/python.exe'
        if not standard.is_file():
            self.skipTest('Standard per-user Python 3.10 fixture not available')
        with tempfile.TemporaryDirectory(prefix='FaceArt discovery ') as temporary:
            root = Path(temporary)
            shutil.copyfile(Path(__file__).resolve().parents[1] / 'Setup-FaceArt.cmd', root / 'Setup-FaceArt.cmd')
            (root / 'scripts').mkdir()
            (root / 'scripts/setup_windows.py').write_text('import sys\nprint("SELECTED="+sys.executable)\n', encoding='utf-8')
            wrong = root / 'wrong python'
            wrong.mkdir()
            # Simulates either an incompatible interpreter or an alias that
            # exits successfully without actually evaluating Python code.
            (wrong / 'python.cmd').write_text(f'@echo off\necho WRONG_PYTHON\nexit /b {candidate_exit}\n')
            environment = os.environ.copy()
            environment['PATH'] = str(wrong) + os.pathsep + str(Path(environment['WINDIR']) / 'System32')
            result = subprocess.run([str(Path(environment['WINDIR']) / 'System32/cmd.exe'), '/d', '/c', 'Setup-FaceArt.cmd', '--dry-run'],
                                    cwd=root, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('SELECTED=' + str(standard), result.stdout)

    def test_wrong_path_version_continues_to_standard_python310(self):
        self.run_launcher(candidate_exit=1)

    def test_alias_without_python_output_continues_to_standard_python310(self):
        self.run_launcher(candidate_exit=0)


if __name__ == '__main__':
    unittest.main()
