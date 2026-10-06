"""Private reference camera worker. No recording or virtual-camera output."""

import base64
import json
import sys
import time
import cv2


def square_crop(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    return frame[
        (h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2
    ].copy()


def main():
    from modules.video_capture import VideoCapturer

    capture = VideoCapturer(int(sys.argv[1]))
    try:
        if not capture.start(1280, 720, 15):
            raise RuntimeError(
                "Cannot open the selected camera. Close other camera apps and retry."
            )
        last = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("The reference camera stopped delivering frames.")
            now = time.monotonic()
            if now - last < 0.15:
                continue
            last = now
            ok, png = cv2.imencode(".png", square_crop(frame))
            if not ok:
                raise RuntimeError("Cannot encode the camera reference.")
            print(
                "REFERENCE:"
                + json.dumps({"frame": base64.b64encode(png).decode("ascii")}),
                flush=True,
            )
    except Exception as error:
        print("REFERENCE:" + json.dumps({"error": str(error)}), flush=True)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
