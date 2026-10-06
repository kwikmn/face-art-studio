"""Private offline/camera comparison workspace; no audio or publishing."""
import isolated_runtime
from pathlib import Path
from datetime import datetime
import json
import time
import traceback
import psutil
import subprocess
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QFileDialog, QCheckBox
from modules.studio.background import BackgroundEffect, BackgroundSettings

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / 'state' / 'matting-lab.json'
PROFILES = {'Baseline: 960x540, GPEN 39': (960,540,True), 'Fast experiment: 640x360, GPEN off': (640,360,False)}

def matched_comparison(raw, protected, alpha):
    old = comparison(protected, alpha)
    h,w = raw.shape[:2]
    canvas = np.concatenate((raw.copy(),old),axis=1)
    for index,title in enumerate(('Raw input (prepared)','Swapped / enhanced','Alpha (whole person)','Final checkerboard')):
        cv2.rectangle(canvas,(index*w,0),((index+1)*w,32),(24,24,24),-1)
        cv2.putText(canvas,title,(index*w+8,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return canvas


def comparison(frame, alpha):
    h, w = frame.shape[:2]
    yy, xx = np.indices((h, w))
    checker = np.where(((xx // 20 + yy // 20) % 2)[..., None], 170, 220).astype(np.uint8)
    checker = np.repeat(checker, 3, axis=2)
    composite = cv2.blendLinear(frame, checker, alpha.astype(np.float32), (1-alpha).astype(np.float32))
    matte = np.repeat((alpha * 255).astype(np.uint8)[..., None], 3, axis=2)
    canvas = np.concatenate((frame, matte, composite), axis=1)
    for x, title in ((0, 'Input (swap fixed if enabled)'), (w, 'Alpha'), (2 * w, 'Composite')):
        cv2.putText(canvas, title, (x + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
        cv2.putText(canvas, title, (x + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return canvas


class Worker(QThread):
    ready = Signal()
    frame = Signal(object)
    status = Signal(str)
    def __init__(self, source, backend, provider, swap_source=None, seconds=0, profile='Baseline: 960x540, GPEN 39'):
        super().__init__()
        self.source, self.backend, self.provider = source, backend, provider
        self.swap_source, self.seconds = swap_source, seconds
        self.profile = profile
        self.latest_stages = None
        self.stop_requested = False
        self.latest = None
        self.report = {}

    def run(self):
        capture = None
        try:
            cv2.setNumThreads(1)
            effect = BackgroundEffect()
            cfg = BackgroundSettings(mode='color', backend=self.backend, provider=self.provider, smoothing=0, softness=0, chair_cleanup=0, tightness=0)
            processor = None
            width,height,gpen = PROFILES[self.profile]
            if self.swap_source:
                from modules.live_face import LiveFaceProcessor
                processor = LiveFaceProcessor(self.swap_source, width=width, height=height, mirror=False)
                processor.capture_stages = True
                processor.prepare()
                processor.set_jaw_feathering(.70)
                # Same fixed swap and enhancement strengths for every matting backend.
                from modules.studio.live_enhancement import LiveGPEN
                if gpen:
                    processor.set_live_enhancement(LiveGPEN(), .39)
            image = None
            if isinstance(self.source, str) and Path(self.source).suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                image = cv2.imread(self.source)
                if image is None:
                    raise ValueError('Cannot decode image')
            else:
                capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW) if isinstance(self.source, int) else cv2.VideoCapture(self.source)
                if not capture.isOpened():
                    raise RuntimeError('Input unavailable; camera may be in use by another app')
                if isinstance(self.source, int):
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    capture.set(cv2.CAP_PROP_FPS, 30)
            fps = capture.get(cv2.CAP_PROP_FPS) if capture is not None else 20
            fps = fps if 1 <= fps <= 60 else 30
            interval = 1 / fps
            self.ready.emit()
            start = time.perf_counter()
            timings, end_to_end, missed, rejected = [], [], 0, 0
            while not self.stop_requested:
                tick = time.perf_counter()
                if self.seconds and tick - start >= self.seconds:
                    break
                if image is not None:
                    raw = image.copy()
                else:
                    ok, raw = capture.read()
                    if not ok:
                        break
                captured_raw = raw.copy()
                from modules.live_pipeline import fit_frame
                raw = fit_frame(raw, width, height)
                protected = raw
                if processor:
                    try:
                        result = processor.process(raw)
                    except ValueError:
                        rejected += 1
                        self.status.emit('Frame rejected: keep one full face visible')
                        self.msleep(10)
                        continue
                    if result.image is None:
                        rejected += 1
                        continue
                    protected = result.image
                t = time.perf_counter()
                effect.apply(protected, cfg)
                elapsed = (time.perf_counter() - t) * 1000
                alpha = effect.last_alpha
                canvas = matched_comparison(raw, protected, alpha)
                self.latest = (protected.copy(), alpha.copy(), canvas.copy())
                self.latest_stages = {'captured_raw':captured_raw, 'prepared_raw':raw.copy(), **(processor.last_stages if processor else {}), 'enhanced':protected.copy()}
                self.frame.emit(canvas)
                duration = (time.perf_counter() - tick) * 1000
                timings.append(elapsed)
                end_to_end.append(duration)
                missed += duration > interval * 1000
                self.status.emit(f'{self.backend} / {self.provider}: matting {elapsed:.1f} ms | frame {duration:.1f} ms | missed budget {missed}/{len(timings)}')
                remaining = interval - (time.perf_counter() - tick)
                if remaining > 0:
                    self.msleep(int(remaining * 1000))
                if image is not None and not self.seconds:
                    break
            telemetry = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            self.report = {'backend': self.backend, 'provider': self.provider, 'frames': len(timings), 'swap_rejections': rejected, 'missed_frame_budget': missed, 'target_fps': fps, 'seconds': time.perf_counter()-start, 'warmup_frames_included': True, 'matting_ms_p50_p95': np.percentile(timings,[50,95]).tolist() if timings else [], 'frame_ms_p50_p95': np.percentile(end_to_end,[50,95]).tolist() if end_to_end else [], 'camera': isinstance(self.source,int), 'swap_enabled': processor is not None, 'rss_mib':psutil.Process().memory_info().rss/2**20,'gpu_used_free_mib_util_pct':telemetry.stdout.strip(),'dropped_capture_frames':'not observable with this sequential capture API'}
            self.report.update(profile=self.profile, resolution=[width,height], GPEN_strength=39 if gpen and processor else 0, jaw_feather=70, stage_capture_overhead_included=True, concurrent_workload='unconfirmed')
            effect.reset()
        except Exception:
            self.report['error'] = traceback.format_exc()
            self.status.emit(self.report['error'])
        finally:
            if capture is not None:
                capture.release()


class Lab(QWidget):
    def __init__(self):
        super().__init__()
        isolated_runtime.configure_fonts(QApplication.instance())
        self.setWindowTitle('FaceArt Studio v0.2 — Matting Lab')
        self.resize(1400, 620)
        self.worker = None
        self.source = str(ROOT / 'media' / 'ludwig.gif')
        self.swap_source = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Private: Raw input | Swapped/enhanced | Alpha | Final. Fast profile loses resolution and GPEN detail. No audio or publishing.'))
        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.backend = QComboBox()
        self.backend.addItems(['mediapipe','rvm','modnet'])
        self.provider = QComboBox()
        self.provider.addItems(['cpu','cuda'])
        self.provider.setCurrentText('cuda')
        controls.addWidget(self.backend)
        controls.addWidget(self.provider)
        self.profile = QComboBox()
        self.profile.addItems(list(PROFILES))
        controls.addWidget(self.profile)
        self.swap = QCheckBox('Face swap (GPEN follows profile)')
        controls.addWidget(self.swap)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        for title, action in [('Open clip/image',self.open_input),('Replacement face',self.open_face),('Start file',self.start_file),('Start webcam',self.start_camera),('Stop',self.stop),('Save comparison',self.save)]:
            button = QPushButton(title)
            button.clicked.connect(action)
            controls.addWidget(button)
        self.camera_index = QComboBox()
        self.camera_index.addItems(['0','1','2'])
        controls.addWidget(self.camera_index)
        self.input_label = QLabel(self.source)
        layout.addWidget(self.input_label)
        self.preview = QLabel('Choose a backend and start. Switching backends stops the current input; start again to reset temporal state.')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview,1)
        self.status = QLabel('Ready')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.backend.currentTextChanged.connect(self.backend_changed)
        self.provider.currentTextChanged.connect(self.stop)
        self.profile.currentTextChanged.connect(self.stop)
        try:
            saved = json.loads(SETTINGS.read_text())
            self.backend.setCurrentText(saved['backend'])
            self.provider.setCurrentText(saved['provider'])
        except (OSError, KeyError, ValueError):
            pass
        self.backend_changed()

    def backend_changed(self,*_):
        self.stop()
        media = self.backend.currentText() == 'mediapipe'
        self.provider.setEnabled(not media)
        if media:
            self.provider.setCurrentText('cpu')

    def open_input(self):
        path,_ = QFileDialog.getOpenFileName(self,'Choose local test input',str(ROOT),'Media (*.mp4 *.mkv *.avi *.gif *.png *.jpg *.jpeg *.webp)')
        if path:
            self.stop()
            self.source = path
            self.input_label.setText(path)

    def open_face(self):
        path,_ = QFileDialog.getOpenFileName(self,'Choose replacement face',str(ROOT),'Images (*.png *.jpg *.jpeg)')
        if path:
            self.swap_source = path
            self.swap.setChecked(True)

    def start_file(self):
        self.start(self.source)

    def start_camera(self):
        self.start(int(self.camera_index.currentText()))

    def start(self, source):
        self.stop()
        self.input_label.setText(f'Camera index {source}' if isinstance(source,int) else str(source))
        self.preview.clear()
        self.preview.setText('Loading isolated models; no output until ready')
        if self.swap.isChecked() and not self.swap_source:
            self.status.setText('Choose a replacement face first')
            return
        self.worker = Worker(source,self.backend.currentText(),self.provider.currentText(),self.swap_source if self.swap.isChecked() else None,profile=self.profile.currentText())
        self.worker.frame.connect(self.show_frame)
        self.worker.status.connect(self.status.setText)
        self.worker.start()
        SETTINGS.write_text(json.dumps({'backend':self.backend.currentText(),'provider':self.provider.currentText()},indent=2))

    def stop(self,*_):
        if self.worker and self.worker.isRunning():
            self.worker.stop_requested = True
            self.worker.wait()  # let current inference finish before destroying GPU state

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

    def save(self):
        # Freeze the worker before reading its matching image/alpha/stage bundle.
        self.stop()
        if not self.worker or self.worker.latest is None:
            return
        out = ROOT / 'state' / 'outputs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        out.mkdir(parents=True,exist_ok=True)
        frame,alpha,canvas = self.worker.latest
        cv2.imwrite(str(out/'comparison.png'),canvas)
        cv2.imwrite(str(out/'alpha.png'),np.rint(alpha*255).astype(np.uint8))
        rgba = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
        rgba[:,:,3] = np.rint(alpha*255).astype(np.uint8)
        cv2.imwrite(str(out/'cutout.png'),rgba)
        for name,image in (self.worker.latest_stages or {}).items():
            cv2.imwrite(str(out/(name+'.png')),image)
        (out/'metrics.json').write_text(json.dumps(self.worker.report,indent=2))
        self.status.setText(f'Saved locally: {out}')

    def closeEvent(self,event):
        self.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication([])
    window = Lab()
    window.show()
    app.exec()
