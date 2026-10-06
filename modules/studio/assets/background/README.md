# MediaPipe landscape segmentation model

`mediapipe.onnx` is bundled from royshil/obs-backgroundremoval revision ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808, data/models/mediapipe.onnx. The source identifies Google LLC and Katsuya Hyodo, with additional redistribution attribution in the adjacent .license file. Retain that file and Apache-2.0.txt with the weights.

Source: https://github.com/royshil/obs-backgroundremoval/blob/ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808/data/models/mediapipe.onnx

SHA-256: 7f785cf032261a07af7b845f891cab30da3f0757c7b362310e089e3aa8e8860a

Input: float32 RGB [1,144,256,3], normalized to [0,1]. Output: [1,144,256,2], foreground probability in channel 1. FaceArt's Python processing/compositing implementation is independent of the OBS plugin code.
