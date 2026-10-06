"""Qt control/status harness only. No real camera/device/audio activation."""
import isolated_runtime
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
from types import SimpleNamespace
import json
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QLabel
from modules.studio.app import Studio,STYLE
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'state'/'outputs'/'candidate5';OUT.mkdir(parents=True,exist_ok=True)
app=QApplication([]);isolated_runtime.configure_fonts(app);app.setStyleSheet(STYLE)
window=Studio(enumerate_devices=False);window.timer.stop()
assert not window.virtual.isChecked();assert not window.virtual.isEnabled()
state=dict(output_enabled=False,output_error=None,status='Live',recent_safe_fps=8.2,processing_p50_ms=120,frame_age_p95_ms=200,source_status='',safe_frames=3,output_device=None)
calls=[]
def enable(value):state['output_enabled']=value;state['output_error']=None;calls.append(value)
window.session=SimpleNamespace(metrics=lambda:state.copy(),preview=lambda:(0,None),enable_output=enable,stop=lambda:None,processor=SimpleNamespace(enhancement_error=None,timings={},background_error=None))
window.virtual.setEnabled(True);window.virtual.setChecked(True);assert calls==[True]
state['output_device']=dict(device='OBS Virtual Camera',backend='obs',width=640,height=360,running=True)
window.last_metrics=0;window._tick();assert 'Sending locally' in window.virtual_status.text()
state.update(output_enabled=False,output_error='Synthetic busy-device error');window.last_metrics=0;window._tick()
assert not window.virtual.isChecked();assert 'Synthetic busy' in window.virtual_status.text()
window.virtual.setChecked(True);window.last_metrics=0;window._tick();assert state['output_enabled'];assert window.notice.text()=='Local virtual camera is sending.'
# Show actual output widgets in a small labelled engineering capture. This is
# a status fixture, not proof of a running webcam or browser integration.
card=QWidget();card.setStyleSheet('QWidget { background:#151d21; }');layout=QVBoxLayout(card)
heading=QLabel('Virtual camera output — v0.2 experiment');layout.addWidget(heading)
layout.addWidget(QLabel('Engineering status fixture. Separate synthetic DirectShow receipt test passed.'))
layout.addWidget(window.virtual);layout.addWidget(window.virtual_status)
layout.addWidget(QLabel('Video: OBS Virtual Camera. Audio: select Voicemod separately in receiving app.'))
card.resize(820,230);card.show();app.processEvents();card.grab().save(str(OUT/'virtual-camera-controls.png'))
window.virtual.setChecked(False);assert calls[-1] is False
window.stop_camera();assert not window.virtual.isChecked();assert not window.virtual.isEnabled()
card.close();window.close();app.processEvents()
(OUT/'ui-control-checks.json').write_text(json.dumps(dict(default_off=True,start_stop=True,actual_device_status_fixture=True,busy_status_retry=True,stop_camera_resets_output=True,hardware_device_activated=False,interactive_acceptance=False),indent=2))
print('Virtual output Qt control/status checks passed.',flush=True)
