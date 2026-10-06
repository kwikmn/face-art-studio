import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pip._internal.configuration import Configuration, kinds
from scripts import setup_windows


class SetupPipIsolationTests(unittest.TestCase):
    def test_null_config_excludes_fabricated_global_and_site_sources(self):
        with tempfile.TemporaryDirectory(prefix='FaceArt pip fixture ') as folder:
            root = Path(folder)
            global_config = root / 'global-pip.ini'
            site_config = root / 'site-pip.ini'
            global_config.write_text('[global]\nextra-index-url = https://example.invalid/global\n', encoding='utf-8')
            site_config.write_text('[global]\nfind-links = https://example.invalid/site\n', encoding='utf-8')
            mapping = {kinds.GLOBAL: [str(global_config)], kinds.SITE: [str(site_config)], kinds.USER: []}
            # Establish the regression using fabricated files only: --isolated
            # still reads both configuration classes without the null override.
            with patch('pip._internal.configuration.get_configuration_files', return_value=mapping), patch.dict(os.environ, {}, clear=True):
                config = Configuration(isolated=True)
                config.load()
                self.assertEqual(config.get_value('global.extra-index-url'), 'https://example.invalid/global')
                self.assertEqual(config.get_value('global.find-links'), 'https://example.invalid/site')
            script = (
                'import os; from unittest.mock import patch; '
                'from pip._internal.configuration import Configuration; '
                f'mapping={mapping!r}; '
                'assert os.environ["PIP_CONFIG_FILE"]==os.devnull; '
                'assert os.environ["PYTHONPATH"]==""; '
                'p=patch("pip._internal.configuration.get_configuration_files", return_value=mapping); '
                'p.start(); c=Configuration(isolated=True); c.load(); '
                'assert dict(c.items())=={}, dict(c.items()); print("PIP_CONFIG_ISOLATED_OK")'
            )
            log = io.StringIO()
            inherited = {'PIP_CONFIG_FILE': str(global_config),
                         'PIP_EXTRA_INDEX_URL': 'https://example.invalid/environment',
                         'PIP_FIND_LINKS': 'https://example.invalid/environment-links',
                         'PYTHONPATH': str(root / 'unrelated-development-code')}
            with patch.object(setup_windows, 'ROOT', root), patch.dict(os.environ, inherited):
                setup_windows.command([sys.executable, '-s', '-c', script], log)
            self.assertIn('PIP_CONFIG_ISOLATED_OK', log.getvalue())


if __name__ == '__main__':
    unittest.main()
