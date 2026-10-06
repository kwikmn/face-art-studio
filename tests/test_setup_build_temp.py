import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import setup_windows


class SetupBuildTempTests(unittest.TestCase):
    def test_deep_project_uses_short_unique_temp_and_preserves_state(self):
        with tempfile.TemporaryDirectory(prefix='fas-test-') as folder:
            root = Path(folder) / ('nested project with spaces ' * 5).rstrip()
            root.mkdir()
            state = root / 'state'
            state.mkdir()
            preserved = state / 'keep.txt'
            preserved.write_text('retain settings/logs/cache', encoding='utf-8')
            log = io.StringIO()
            script = ('import os; from pathlib import Path; '
                      'p=Path(os.environ["TMP"]); '
                      'assert os.environ["TEMP"]==str(p); '
                      f'assert len(str(p))<={setup_windows.MAX_BUILD_TEMP_PATH}; '
                      'assert p.is_dir(); assert p.name.startswith("fas-"); '
                      'assert p.resolve() not in Path.cwd().resolve().parents; '
                      '(p/"compiler-output.obj").write_text("synthetic"); print("BUILD_TEMP="+str(p))')
            with patch.object(setup_windows, 'ROOT', root):
                setup_windows.command([sys.executable, '-s', '-c', script], log)
            build_temp = Path(log.getvalue().split('BUILD_TEMP=', 1)[1].strip())
            self.assertNotIn(root, build_temp.parents)
            self.assertFalse(build_temp.exists())
            self.assertEqual(preserved.read_text(encoding='utf-8'), 'retain settings/logs/cache')
            self.assertFalse((state / 'setup-temp').exists())

    def test_long_temp_base_stops_before_spawning_package_command(self):
        with tempfile.TemporaryDirectory(prefix='fas-test-') as folder:
            base = Path(folder) / ('long temp ' * 8)
            base.mkdir()
            with patch.object(setup_windows.sys, 'platform', 'linux'), patch.object(setup_windows.tempfile, 'gettempdir', return_value=str(base)), patch.object(setup_windows.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(RuntimeError, 'build temp path is too long'):
                    setup_windows.command([sys.executable, '-c', 'raise SystemExit(99)'], io.StringIO())
                popen.assert_not_called()

    def test_failed_command_cleans_owned_temp_and_reports_actual_remedy(self):
        with tempfile.TemporaryDirectory(prefix='fas-test-') as folder:
            root = Path(folder)
            log = io.StringIO()
            script = 'import os; print("BUILD_TEMP="+os.environ["TMP"], flush=True); raise SystemExit(1)'
            with patch.object(setup_windows, 'ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'reinstalling them is not the first remedy'):
                    setup_windows.command([sys.executable, '-s', '-c', script], log)
            build_temp = Path(log.getvalue().split('BUILD_TEMP=', 1)[1].strip())
            self.assertFalse(build_temp.exists())


if __name__ == '__main__':
    unittest.main()
