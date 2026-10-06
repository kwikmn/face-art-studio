"""One Windows GUI owner per login session; prevents virtual-camera collisions."""
import atexit
import sys

_handle = None


def acquire_gui_instance(window_title=None):
    global _handle
    if sys.platform != "win32" or _handle is not None:
        return True
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, False, "Local\\FaceArtStudio.v02.GUI")
    already_exists = ctypes.get_last_error() == 183
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if already_exists:
        kernel.CloseHandle(handle)
        from modules import metadata
        user = ctypes.WinDLL("user32", use_last_error=True)
        user.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user.FindWindowW.restype = wintypes.HWND
        user.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user.SetForegroundWindow.argtypes = [wintypes.HWND]
        title = f"{metadata.name} {metadata.version} {metadata.edition}"
        window = user.FindWindowW(None, window_title or title)
        if not window:
            window = user.FindWindowW(None, 'FaceArt Studio v0.2 experimental')
        if window:
            user.ShowWindow(window, 9)
            user.SetForegroundWindow(window)
        print("DLC is already running; keeping the existing instance.", flush=True)
        return False
    _handle = handle
    atexit.register(kernel.CloseHandle, handle)
    return True
