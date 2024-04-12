import torch
import ctypes

from config import use_trt_flag, CLASSIFY_MODEL_PATH, CLASSIFY_TRT_PATH, COIN_VALUE_MODEL_PATH, \
    DETECT_MODEL_PATH, COIN_VALUE_TRT_PATH, PLUGIN_LIBRARY, DETECT_TRT_PATH
from views.predict_impl import TensorRT, Onnx, YoLov5TRT, TinyTensorRT

device_type = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device_type)
device = torch.device(device_type)

if use_trt_flag:
    session_back = TensorRT(CLASSIFY_MODEL_PATH, CLASSIFY_TRT_PATH)
    session_value = TensorRT(COIN_VALUE_MODEL_PATH, COIN_VALUE_TRT_PATH)

    ctypes.CDLL(PLUGIN_LIBRARY)
    session_d = YoLov5TRT(DETECT_TRT_PATH)
else:
    session_back = Onnx(CLASSIFY_MODEL_PATH)
    session_value = Onnx(COIN_VALUE_MODEL_PATH)
    session_d = Onnx(DETECT_MODEL_PATH)
