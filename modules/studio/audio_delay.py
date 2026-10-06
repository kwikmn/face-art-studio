"""Bounded PCM delay for a streaming microphone (no inference or encoding)."""


class PCMDelay:
    def __init__(self, sample_rate, frame_bytes, delay_ms):
        if (
            isinstance(delay_ms, bool)
            or not isinstance(delay_ms, int)
            or not 0 <= delay_ms <= 2000
        ):
            raise ValueError("Audio delay must be a whole number from 0 to 2000 ms")
        self.frame_bytes = frame_bytes
        self.data = bytearray(sample_rate * delay_ms // 1000 * frame_bytes)
        self.limit = len(self.data) + sample_rate * frame_bytes // 2

    def push(self, data):
        if len(data) % self.frame_bytes:
            raise ValueError("Incomplete PCM frame")
        if len(self.data) + len(data) > self.limit:
            raise RuntimeError(
                "Streaming audio could not keep up. Stop and restart audio."
            )
        self.data.extend(data)

    def take(self, count):
        count = min(count // self.frame_bytes * self.frame_bytes, len(self.data))
        result = bytes(self.data[:count])
        del self.data[:count]
        return result

    def prepend(self, data):
        self.data[:0] = data
