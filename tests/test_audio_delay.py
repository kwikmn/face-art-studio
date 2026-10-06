import unittest
from unittest.mock import patch
from modules.studio.audio_delay import PCMDelay
from modules.studio.recording import Recorder


class AudioDelayTests(unittest.TestCase):
    def test_exact_delay_all_channels_across_irregular_blocks(self):
        for ms in (0, 10, 1000, 2000):
            delay = PCMDelay(48000, 4, ms)
            payload = bytes(range(256)) * 4
            delay.push(payload[:100])
            delay.push(payload[100:])
            result = b""
            while delay.data:
                result += delay.take(123)
            self.assertEqual(result, bytes(48000 * ms // 1000 * 4) + payload)

    def test_bounded_backpressure_does_not_silently_drop_audio(self):
        delay = PCMDelay(1000, 2, 1000)
        delay.push(bytes(1000))
        with self.assertRaises(RuntimeError):
            delay.push(bytes(2))
        self.assertEqual(len(delay.data), 3000)

    def test_short_writes_keep_order(self):
        delay = PCMDelay(48000, 2, 0)
        delay.push(b"12345678")
        block = delay.take(6)
        delay.prepend(block[2:])
        self.assertEqual(delay.take(100), b"345678")

    def test_rejects_invalid_delays(self):
        for value in (-1, 2001, 0.5, True, "1000"):
            with self.assertRaises(ValueError):
                PCMDelay(48000, 4, value)
            with self.assertRaises(ValueError):
                Recorder(lambda: None, 80, 60, "unused.mkv", audio_delay_ms=value)

    def test_recorder_normalizes_timestamps_then_delays_all_audio_channels(self):
        with patch("modules.studio.recording.executable", return_value="ffmpeg"):
            recorder = Recorder(
                lambda: None, 80, 60, "unused.mkv", "Test mic", audio_delay_ms=1000
            )
            command = recorder._command()
            self.assertEqual(
                command[command.index("-af") + 1],
                "aresample=async=1000:first_pts=0,adelay=delays=1000:all=1",
            )
            self.assertIn("FACEART_AUDIO_DELAY_MS=1000", command)
            recorder.audio_delay_ms = 0
            self.assertNotIn("adelay", " ".join(recorder._command()))
            recorder.microphone = ""
            self.assertNotIn("-af", recorder._command())


if __name__ == "__main__":
    unittest.main()
