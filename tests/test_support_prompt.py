import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from PySide6.QtWidgets import QApplication
from modules.studio.app import Studio, SUPPORT_MESSAGE, SUPPORT_URL


class SupportPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = Path(self.temporary.name) / 'settings.json'
        self.patches = [
            patch('modules.studio.app.SETTINGS', self.settings),
            patch.object(Studio, 'refresh_audio_outputs'),
            patch.object(Studio, 'refresh_devices'),
            patch.object(Studio, 'refresh_microphones'),
            patch('modules.studio.app.QDesktopServices.openUrl', return_value=True),
        ]
        for item in self.patches:
            item.start()
        self.open_url = self.patches[-1].target.openUrl
        self.windows = []

    def window(self):
        window = Studio(enumerate_devices=False)
        window.timer.stop()
        self.windows.append(window)
        return window

    def tearDown(self):
        for window in self.windows:
            window.session = window.recorder = window.live_audio = window.projection = None
            window.timer.stop()
            window._shutdown()
            window.hide()
            window.deleteLater()
        self.app.processEvents()
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def test_idle_card_never_opens_link_automatically(self):
        window = self.window()
        self.assertFalse(window.support_prompt.isHidden())
        self.assertEqual(window.support_message.text(), SUPPORT_MESSAGE)
        self.open_url.assert_not_called()

    def test_click_opens_exact_link_once_and_remembers_choice(self):
        window = self.window()
        window.support_button.click()
        self.open_url.assert_called_once()
        self.assertEqual(self.open_url.call_args.args[0].toString(), SUPPORT_URL)
        self.assertEqual(SUPPORT_URL, 'https://paypal.me/WikmanKarl')
        self.assertTrue(json.loads(self.settings.read_text())['support_prompt_dismissed'])
        window.support_button.click()
        self.open_url.assert_called_once()
        self.assertTrue(self.window().support_prompt.isHidden())

    def test_dismissal_survives_restart_without_opening_link(self):
        window = self.window()
        window.support_dismiss.click()
        self.assertTrue(window.support_prompt.isHidden())
        self.assertTrue(self.window().support_prompt.isHidden())
        self.open_url.assert_not_called()

    def test_live_or_busy_work_hides_card_and_blocks_link(self):
        for field in ['session', 'recorder', 'live_audio', 'job_busy', 'record_saving', 'projection']:
            with self.subTest(field=field):
                window = self.window()
                setattr(window, field, True)
                # Guard also blocks a click racing with the UI update.
                window.support_button.click()
                self.open_url.assert_not_called()
                window._update_support_prompt()
                self.assertTrue(window.support_prompt.isHidden())
                setattr(window, field, None)
                window._update_support_prompt()
                self.assertTrue(window.support_prompt.isHidden())

    def test_browser_failure_stays_non_modal_and_does_not_dismiss(self):
        self.open_url.return_value = False
        window = self.window()
        window.support_button.click()
        self.assertFalse(window.support_prompt.isHidden())
        self.assertNotIn('support_prompt_dismissed', window.settings)
        self.assertIn('README.md', window.support_message.text())


if __name__ == '__main__':
    unittest.main()
