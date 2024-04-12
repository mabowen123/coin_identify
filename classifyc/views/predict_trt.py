import numpy as np
import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from .base import PredictBase
from .process import DetectProcess, ClsProcess, ValueProcess,ClsProcess2


logger = trt.Logger(trt.Logger.WARNING)


class TrtPredictBase(PredictBase):
    def __init__(self, model_path):
        runtime = trt.Runtime(logger)
        with open(model_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        num_bindings = self.context.engine.num_bindings

        inputs_bindings = [i for i in range(num_bindings) if self.context.engine.binding_is_input(i)]
        outputs_bindings = [i for i in range(num_bindings) if not self.context.engine.binding_is_input(i)]

        self.input_names = {binding_idx: self.engine.get_binding_name(binding_idx) for binding_idx in inputs_bindings}
        self.output_names = {binding_idx: self.engine.get_binding_name(binding_idx) for binding_idx in outputs_bindings}
        self.input_shapes = {binding_idx: self.engine.get_binding_shape(binding_idx) for binding_idx in inputs_bindings}
        self.output_shapes = {binding_idx: self.engine.get_binding_shape(binding_idx) for binding_idx in
                              outputs_bindings}

        self.host_inputs = [
            cuda.pagelocked_empty(
                trt.volume(self.context.get_binding_shape(binding_idx)),
                trt.nptype(self.engine.get_binding_dtype(binding_idx))
            ) for binding_idx in inputs_bindings
        ]
        self.host_outputs = [
            cuda.pagelocked_empty(
                trt.volume(self.context.get_binding_shape(binding_idx)),
                trt.nptype(self.engine.get_binding_dtype(binding_idx))
            ) for binding_idx in outputs_bindings
        ]

        self.device_inputs = [cuda.mem_alloc(hi.nbytes) for hi in self.host_inputs]
        self.device_outputs = [cuda.mem_alloc(ho.nbytes) for ho in self.host_outputs]
        self.bindings = [int(i) for i in self.device_inputs] + [int(i) for i in self.device_outputs]
        self.stream = cuda.Stream()

    def predict(self, *array):
        [np.copyto(hi, arr.ravel()) for hi, arr in zip(self.host_inputs, array)]
        [cuda.memcpy_htod_async(di, hi, self.stream) for di, hi in zip(self.device_inputs, self.host_inputs)]
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(ho, do, self.stream) for ho, do in zip(self.host_outputs, self.device_outputs)]
        self.stream.synchronize()
        return [out.reshape(self.output_shapes[shape]) for shape, out in zip(self.output_shapes, self.host_outputs)]


class DetectTrt(DetectProcess, TrtPredictBase):
    def __init__(self, trt_path, short_size, conf_thresh):
        DetectProcess.__init__(self, short_size, conf_thresh)
        TrtPredictBase.__init__(self, trt_path)

    def __call__(self, img):
        detect_processed_img = self.preprocess(img)
        detect_output_array = self.predict(detect_processed_img)[3]
        detect_output_array = torch.from_numpy(detect_output_array)
        detect_objects = self.post_process(detect_output_array, img.shape[:2])
        return detect_objects


class ClsTrt(ClsProcess2, TrtPredictBase):
    def __init__(self, trt_path):
        ClsProcess2.__init__(self)
        TrtPredictBase.__init__(self, trt_path)

    def __call__(self, img):
        cls_processed_img = self.preprocess(img)
        print(cls_processed_img)
        cls_output_array = self.predict(cls_processed_img)[0]
        print(cls_output_array)
        cls_label = self.post_process(cls_output_array)
        return cls_label


class ValueTrt(ValueProcess, TrtPredictBase):
    def __init__(self, trt_path):
        ValueProcess.__init__(self)
        TrtPredictBase.__init__(self, trt_path)

    def __call__(self, img):
        cls_processed_img = self.preprocess(img)
        cls_output_array = self.predict(cls_processed_img)[0]
        cls_label = self.post_process(cls_output_array)
        return cls_label
