"""Local protected webcam runner and timing-only benchmark.

Examples:
  python run_protected_camera.py --list-cameras
  python run_protected_camera.py --source FACE.png --camera 0
  python run_protected_camera.py --source FACE.png --input-image TEST.png --seconds 30 --no-camera-output
No camera images or audio are recorded. Ctrl+C stops with a final slate.
"""
import argparse
import json
from pathlib import Path
import time

from modules.live_pipeline import LiveConfig, LivePipeline
from modules.live_face import LiveFaceProcessor
from modules.virtual_camera import VirtualCameraOutput


class ImageCapture:
    def __init__(self, path):
        from modules import imread_unicode
        self.frame = imread_unicode(path)
        if self.frame is None:
            raise ValueError('Cannot read test image')

    def start(self, width, height, fps):
        self.period = 1 / fps
        return True

    def read(self):
        time.sleep(self.period)
        return True, self.frame.copy()

    def release(self):
        pass


class NullOutput:
    def send(self, frame):
        pass

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--list-cameras', action='store_true')
    parser.add_argument('--input-image')
    parser.add_argument('--height', type=int, choices=[540, 720], default=720)
    parser.add_argument('--seconds', type=float, default=0, help='Run duration including startup; 0 runs until Ctrl+C')
    parser.add_argument('--no-camera-output', action='store_true')
    parser.add_argument('--stats-file', help='JSON metrics only; contains no images or audio')
    args = parser.parse_args()
    if args.list_cameras:
        from pygrabber.dshow_graph import FilterGraph
        for index, name in enumerate(FilterGraph().get_input_devices()):
            print(f'{index}: {name}')
        return
    if not args.source:
        parser.error('--source is required')
    from modules.video_capture import VideoCapturer
    import psutil
    config = LiveConfig(width=1280 if args.height == 720 else 960, height=args.height)
    capture = ImageCapture(args.input_image) if args.input_image else VideoCapturer(args.camera)
    processor = LiveFaceProcessor(args.source, width=config.width, height=config.height)
    output = NullOutput() if args.no_camera_output else VirtualCameraOutput()
    session = LivePipeline(capture, processor, output, config)
    process = psutil.Process()
    process.cpu_percent()
    psutil.cpu_percent(percpu=True)
    voices = []
    for proc in psutil.process_iter(['name']):
        if (proc.info['name'] or '').lower() == 'voicemod.exe':
            proc.cpu_percent()
            voices.append(proc)
    samples = []
    session.start()
    started = time.perf_counter()
    try:
        while not args.seconds or time.perf_counter() - started < args.seconds:
            time.sleep(1)
            voice_cpu = 0
            for proc in voices:
                try:
                    voice_cpu += proc.cpu_percent()
                except psutil.Error:
                    pass
            metrics = session.metrics()
            metrics.update({'cpu_process_percent_one_core_100': process.cpu_percent(),
                            'cpu_voicemod_percent_one_core_100': voice_cpu,
                            'cpu_system_per_core_percent': psutil.cpu_percent(percpu=True),
                            'rss_mb': round(process.memory_info().rss / 1024**2, 1),
                            'stage_ms': dict(processor.timings)})
            samples.append(metrics)
            print(json.dumps(metrics), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stopped = session.stop()
        report = {'configuration': vars(config), 'providers': processor.provider_report,
                  'final': session.metrics(), 'workers_stopped': stopped, 'samples': samples}
        if args.stats_file:
            path = Path(args.stats_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'final': report['final'], 'workers_stopped': stopped}), flush=True)


if __name__ == '__main__':
    main()
