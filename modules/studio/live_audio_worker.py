"""Isolated Qt PCM route to a virtual audio cable; never chooses default devices."""

import json
import sys
import time
from PySide6.QtCore import QCoreApplication, QTimer, Qt
from PySide6.QtMultimedia import (
    QMediaDevices,
    QAudioFormat,
    QAudioSource,
    QAudioSink,
    QtAudio,
)
from modules.studio.audio_delay import PCMDelay


def emit(**message):
    print("AUDIO:" + json.dumps(message), flush=True)


def main():
    app = QCoreApplication(sys.argv)
    request = json.loads(sys.argv[1])
    sources = [
        d
        for d in QMediaDevices.audioInputs()
        if d.description() == request["microphone"]
    ]
    outputs = [
        d
        for d in QMediaDevices.audioOutputs()
        if bytes(d.id()).hex() == request["output"]
    ]
    if len(sources) != 1 or len(outputs) != 1:
        raise RuntimeError(
            "Selected audio device is unavailable or ambiguous. Refresh and select again."
        )
    source_device, output_device = sources[0], outputs[0]
    # Only a dedicated cable is suitable: Voicemod's own playback endpoint can loop back.
    if "cable" not in output_device.description().lower():
        raise RuntimeError("Choose a dedicated virtual audio cable output.")
    if "cable" in source_device.description().lower():
        raise RuntimeError(
            "Choose the original microphone or voice changer, not the cable return."
        )
    fmt = None
    for rate in (48000, 44100):
        for channels in (2, 1):
            candidate = QAudioFormat()
            candidate.setSampleRate(rate)
            candidate.setChannelCount(channels)
            candidate.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            if source_device.isFormatSupported(
                candidate
            ) and output_device.isFormatSupported(candidate):
                fmt = candidate
                break
        if fmt:
            break
    if fmt is None:
        raise RuntimeError(
            "Input and cable need a shared 48 kHz or 44.1 kHz PCM format."
        )
    source = QAudioSource(source_device, fmt)
    sink = QAudioSink(output_device, fmt)
    byte_rate = fmt.sampleRate() * fmt.bytesPerFrame()
    source.setBufferSize(byte_rate // 10)
    sink.setBufferSize(byte_rate // 25)
    reader, writer = source.start(), sink.start()
    if reader is None or writer is None:
        raise RuntimeError("Cannot open the selected microphone or cable.")
    delay = PCMDelay(fmt.sampleRate(), fmt.bytesPerFrame(), request["delay_ms"])
    last_input = time.monotonic()
    announced = False
    timer = QTimer()
    timer.setTimerType(Qt.TimerType.PreciseTimer)

    def pump():
        nonlocal last_input, announced
        try:
            input_error, output_error = source.error(), sink.error()
            if input_error != QtAudio.Error.NoError or output_error not in (
                QtAudio.Error.NoError,
                QtAudio.Error.UnderrunError,
            ):
                raise RuntimeError(
                    f"Audio device failed: input {input_error.name}, output {output_error.name}"
                )
            data = bytes(reader.readAll())
            if data:
                last_input = time.monotonic()
                delay.push(data)
                if not announced:
                    announced = True
                    emit(
                        running=True,
                        sample_rate=fmt.sampleRate(),
                        delay_ms=request["delay_ms"],
                    )
            if time.monotonic() - last_input > 2:
                raise RuntimeError(
                    "Microphone stopped delivering audio. No substitute input was selected."
                )
            if announced:
                block = delay.take(sink.bytesFree())
                if block:
                    written = writer.write(block)
                    if written < 0:
                        raise RuntimeError(
                            "Virtual audio cable stopped accepting audio."
                        )
                    delay.prepend(block[written:])
        except Exception as error:
            emit(error=str(error))
            timer.stop()
            app.exit(1)

    timer.timeout.connect(pump)
    timer.start(5)
    try:
        return app.exec()
    finally:
        source.stop()
        sink.reset()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        emit(error=str(error))
        sys.exit(1)
