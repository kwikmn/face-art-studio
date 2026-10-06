import unittest
from unittest.mock import patch
import numpy as np
from modules.live_pipeline import LivePipeline, LiveConfig


class Clock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now


class Stop:
    def __init__(self, clock):
        self.clock = clock
        self.waits = []

    def is_set(self):
        return len(self.waits) >= 3

    def wait(self, delay):
        self.waits.append(delay)
        self.clock.now += delay + (0.017 if len(self.waits) == 1 else 0.0)


class Lock:
    def __init__(self, clock, delay):
        self.clock = clock
        self.delay = delay

    def __enter__(self):
        self.clock.now += self.delay
        self.delay = 0.0

    def __exit__(self, *args):
        pass


class Output:
    def __init__(self, clock):
        self.clock = clock
        self.frames = []

    def send(self, frame):
        self.frames.append((self.clock.now, frame.copy()))

    def close(self):
        pass


class PublisherDelays(unittest.TestCase):
    def run_case(self, lock_delay=0.0):
        clock = Clock()
        out = Output(clock)
        pipeline = LivePipeline(
            None,
            None,
            out,
            LiveConfig(width=20, height=20, fps=30, hold_last_good=0.75),
        )
        pipeline.enable_output(True)  # Fake output only.
        pipeline._stop = Stop(clock)
        pipeline._lock = Lock(clock, lock_delay)
        pipeline._safe = (np.full((20, 20, 3), 88, np.uint8), 0.0)
        with (
            patch("modules.live_pipeline.time.perf_counter", clock.read),
            patch("modules.live_pipeline.sys.platform", "test"),
        ):
            pipeline._publish()
        return pipeline, out

    def test_delayed_wake_drops_tick_without_catchup(self):
        pipeline, out = self.run_case()
        times = [x[0] for x in out.frames[:-1]]
        self.assertAlmostEqual(times[1], 1 / 30 + 0.017)
        self.assertGreaterEqual(min(np.diff(times)), 1 / 30 - 1e-12)
        # Stop's final safety slate intentionally bypasses normal cadence.
        np.testing.assert_array_equal(out.frames[-1][1], pipeline.slate)

    def test_lock_delay_rechecks_freshness(self):
        pipeline, out = self.run_case(0.8)
        np.testing.assert_array_equal(out.frames[0][1], pipeline.slate)
        self.assertGreaterEqual(out.frames[1][0] - out.frames[0][0], 1 / 30)


if __name__ == "__main__":
    unittest.main()
