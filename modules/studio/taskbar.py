"""Windows taskbar identity for the Python-hosted FaceArt application."""

import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys
import uuid

APP_ID = "FaceArt.Studio.Experimental.v02"
ROOT = Path(__file__).resolve().parents[2]
ICON = ROOT / "modules" / "studio" / "assets" / "faceart.ico"


class GUID(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 16)]

    def __init__(self, value):
        super().__init__()
        self.data[:] = uuid.UUID(value).bytes_le


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", wintypes.DWORD)]


class Value(ctypes.Union):
    _fields_ = [("text", ctypes.c_wchar_p), ("storage", ctypes.c_ubyte * 16)]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("reserved", ctypes.c_ushort * 3),
        ("value", Value),
    ]


def _check(result):
    if result < 0:
        raise OSError(
            f"Windows taskbar property failed: HRESULT 0x{result & 0xFFFFFFFF:08x}"
        )


def _method(store, index, *arguments):
    table = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *arguments)(table[index])


def _properties(hwnd=None, shortcut=None, read=False):
    shell = ctypes.WinDLL("shell32")
    iid = GUID("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99")
    store = ctypes.c_void_p()
    if shortcut is not None:
        get_store = shell.SHGetPropertyStoreFromParsingName
        get_store.argtypes = [
            wintypes.LPCWSTR,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        _check(
            get_store(
                str(shortcut),
                None,
                0 if read else 2,
                ctypes.byref(iid),
                ctypes.byref(store),
            )
        )
    else:
        get_store = shell.SHGetPropertyStoreForWindow
        get_store.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        _check(get_store(hwnd, ctypes.byref(iid), ctypes.byref(store)))
    fmtid = GUID("9f4c2855-9f79-4b39-a8d0-e1d42de1d5f3")
    executable = ROOT / "venv" / "Scripts" / "pythonw.exe"
    # Set relaunch information before ID: the ID notifies Explorer to refresh.
    values = {
        2: subprocess.list2cmdline([str(executable), str(ROOT / "run_studio.py")]),
        3: f"{ICON},0",
        4: "FaceArt Studio",
        5: APP_ID,
    }
    result = {}
    try:
        for pid, text in values.items():
            key = PROPERTYKEY(fmtid, pid)
            variant = PROPVARIANT()
            if read:
                _check(
                    _method(
                        store,
                        5,
                        ctypes.POINTER(PROPERTYKEY),
                        ctypes.POINTER(PROPVARIANT),
                    )(store, ctypes.byref(key), ctypes.byref(variant))
                )
                try:
                    result[pid] = variant.value.text if variant.vt == 31 else None
                finally:
                    ctypes.OleDLL("ole32").PropVariantClear(ctypes.byref(variant))
            else:
                variant.vt = 31  # VT_LPWSTR; SetValue copies the string.
                variant.value.text = text
                _check(
                    _method(
                        store,
                        6,
                        ctypes.POINTER(PROPERTYKEY),
                        ctypes.POINTER(PROPVARIANT),
                    )(store, ctypes.byref(key), ctypes.byref(variant))
                )
        if not read:
            _check(_method(store, 7)(store))
        return result
    finally:
        _method(store, 2)(store)


def configure_process():
    if sys.platform == "win32":
        set_id = ctypes.WinDLL("shell32").SetCurrentProcessExplicitAppUserModelID
        set_id.argtypes = [wintypes.LPCWSTR]
        _check(set_id(APP_ID))


def configure_window(window):
    if sys.platform == "win32":
        _properties(hwnd=int(window.winId()))
