import torch
import onnxruntime
from .base import PredictBase
from .process import DetectProcess, ClsProcess, ClsProcess3, ValueProcess,DetectProcessOld


class OnnxPredictBase(PredictBase):
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, array):
        output_numpy_array = self.session.run([self.output_name], {self.input_name: array})[0]
        return output_numpy_array

    
class OnnxPredictBaseOld(PredictBase):
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_tensor):
        output_numpy_array = self.model_session.run([self.output_name], {self.input_name: image_tensor})[0]
        print('begin predict_old_feature 6', output_numpy_array)
        output_numpy_array = torch.from_numpy(output_numpy_array)
        return output_numpy_array
    
    
class OnnxMulPredictBase(PredictBase):
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [i.name for i in self.session.get_outputs()]

    def predict(self, *array):
        output_numpy_array = self.session.run(self.output_names, dict(zip(self.input_names, array)))
        return output_numpy_array


class DetectOnnx(DetectProcess, OnnxPredictBase):
    def __init__(self, onnx_path, short_size, conf_thresh):
        DetectProcess.__init__(self, short_size, conf_thresh)
        OnnxPredictBase.__init__(self, onnx_path)

    def __call__(self, img):
        detect_processed_img = self.preprocess(img)
        detect_output_array = self.predict(detect_processed_img)
        detect_output_array = torch.from_numpy(detect_output_array)
        detect_objects = self.post_process(detect_output_array, img.shape[:2])
        return detect_objects


class DetectOnnx2(PredictBase,DetectProcessOld):
    def __init__(self, model_path, short_size, conf_thresh):
        self.model_session = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.model_session.get_inputs()[0].name
        self.output_name = self.model_session.get_outputs()[0].name
        self.conf_thresh = conf_thresh
        self.short_size = short_size

    def __call__(self, img):
        detect_processed_img, ori_shape = self.preprocess(img)
        detect_output_array = self.predict(detect_processed_img)
        detect_objects = self.post_process(detect_output_array, ori_shape)
        return detect_objects


    def predict(self, image_tensor):
        output_numpy_array = self.model_session.run([self.output_name], {self.input_name: image_tensor})[0]
        output_numpy_array = torch.from_numpy(output_numpy_array)
        return output_numpy_array

# 对应img_size=416的版本
class ClsOnnx(ClsProcess, OnnxPredictBase):
    def __init__(self, onnx_path):
        ClsProcess.__init__(self)
        OnnxPredictBase.__init__(self, onnx_path)

    def __call__(self, img):
        cls_processed_img = self.preprocess(img)
        cls_output_array = self.predict(cls_processed_img)
        cls_label = self.post_process(cls_output_array)
        # print("cls_label topk_probs is:",cls_label[0], 'cls_label topk_indices is:', cls_label[1],cls_label)
        return cls_label[0], cls_label[1]

# 对应img_size=627的版本
class ClsOnnx3(ClsProcess3, OnnxPredictBase):
    def __init__(self, onnx_path):
        ClsProcess3.__init__(self)
        OnnxPredictBase.__init__(self, onnx_path)

    def __call__(self, img):
        cls_processed_img = self.preprocess(img)
        cls_output_array = self.predict(cls_processed_img)
        cls_label = self.post_process(cls_output_array)
        # print("cls_label topk_probs is:",cls_label[0], 'cls_label topk_indices is:', cls_label[1],cls_label)
        return cls_label[0], cls_label[1]
    
class ValueOnnx(ValueProcess, OnnxPredictBase):
    def __init__(self, onnx_path):
        ValueProcess.__init__(self)
        OnnxPredictBase.__init__(self, onnx_path)

    def __call__(self, img):
        cls_processed_img = self.preprocess(img)
        cls_output_array = self.predict(cls_processed_img)
        cls_label = self.post_process(cls_output_array)
        return cls_label