"""Virtual camera output owned exclusively by the live publisher thread."""

import cv2
import numpy as np
import os


def check_obs_available():
    """Read-only producer check; never open/overwrite an existing video queue."""
    if os.name != 'nt':
        raise RuntimeError('This output profile requires the installed Windows OBS backend')
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenFileMappingW.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_wchar_p]
    kernel.OpenFileMappingW.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenFileMappingW(4,False,'OBSVirtualCamVideo')
    if handle:
        kernel.CloseHandle(handle)
        raise RuntimeError('OBS Virtual Camera is already owned by another producer or receiver session. Stop its virtual-camera output yourself, or close its stale local receiver, then retry. FaceArt will not take it over.')
    error=ctypes.get_last_error()
    if error != 2:
        raise RuntimeError(f'Cannot safely verify OBS Virtual Camera ownership (Windows error {error}); output remains off')


class VirtualCameraOutput:
    def __init__(self):
        self._camera = None
        self._info={'backend':'obs','device':None,'running':False,'frames_sent':0}

    def info(self):
        return self._info.copy()

    def send(self, frame):
        # Import lazily so preview remains usable without a virtual-camera backend.
        import pyvirtualcam

        if self._camera is None:
            check_obs_available()
            height, width = frame.shape[:2]
            self._camera = pyvirtualcam.Camera(
                width=width, height=height, fps=30, fmt=pyvirtualcam.PixelFormat.BGR,
                backend='obs', device='OBS Virtual Camera'
            )
            self._info={'backend':'obs','device':self._camera.device,'running':True,
                        'width':width,'height':height,'nominal_fps':30,'frames_sent':0}
        camera = self._camera
        if frame.shape[:2] != (camera.height, camera.width):
            frame = cv2.resize(frame, (camera.width, camera.height))
        camera.send(np.ascontiguousarray(frame))
        self._info={**self._info,'frames_sent':self._info['frames_sent']+1}
        # The independent publisher supplies pacing; send never sleeps.

    def close(self):
        camera, self._camera = self._camera, None
        self._info={**self._info,'running':False}
        if camera is not None:
            try:
                camera.close()
            except Exception as error:
                print(f"Virtual Camera close error: {error}")
