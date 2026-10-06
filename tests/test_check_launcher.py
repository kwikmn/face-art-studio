import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == 'win32', 'Windows CMD launcher regression')
class CheckLauncherTests(unittest.TestCase):
    def test_inherited_pythonpath_is_excluded(self):
        with tempfile.TemporaryDirectory(prefix='FaceArt check ') as folder:
            root = Path(folder)
            shutil.copyfile(Path(__file__).resolve().parents[1] / 'Check-FaceArt.cmd', root / 'Check-FaceArt.cmd')
            scripts = root / 'scripts'
            scripts.mkdir()
            runtime = root / 'venv/Scripts'
            runtime.mkdir(parents=True)
            # A small interpreter copy plus the existing base runtime gives the
            # fixture a real .exe at the launcher's required path, without pip
            # or a new environment installation.
            shutil.copyfile(getattr(sys, '_base_executable', sys.executable), runtime / 'python.exe')
            poison = root / 'unrelated source'
            poison.mkdir()
            (poison / 'faceart_fixture_poison.py').write_text('raise RuntimeError("Inherited development import")\n', encoding='utf-8')
            (scripts / 'check_install.py').write_text(
                'import os, sys\n'
                'assert os.environ.get("PYTHONPATH", "")==""\n'
                'assert sys.flags.no_user_site and sys.dont_write_bytecode\n'
                'try:\n import faceart_fixture_poison\n'
                'except ModuleNotFoundError:\n print("CHECK_ISOLATED_OK")\n'
                'else:\n raise AssertionError("Unexpected inherited source")\n', encoding='utf-8')
            environment = os.environ.copy()
            environment.update(PYTHONPATH=str(poison), PYTHONHOME=sys.base_prefix,
                               PATH=sys.base_prefix + os.pathsep + environment.get('PATH', ''))
            result = subprocess.run([str(Path(environment['WINDIR']) / 'System32/cmd.exe'), '/d', '/c', 'Check-FaceArt.cmd'],
                                    cwd=root, env=environment, input='\n', capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('CHECK_ISOLATED_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
