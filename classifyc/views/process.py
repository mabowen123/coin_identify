
import cv2
import time
import torch
import torchvision
import numpy as np
from .base import ProcessBase
import torchvision.transforms as transforms


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.01, iou_thres=0.5, merge=False, classes=None,
                        agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    device_type = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # print('xi, x, ', xi, x,xc[xi])
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


class DetectProcess(ProcessBase):
    def __init__(self, short_size=640, conf_thresh=0.01):
        self.short_size = short_size
        self.conf_thresh = conf_thresh

    def preprocess(self, img):
        img = letterbox(img, new_shape=(self.short_size, self.short_size))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def post_process(self, output_array, ori_shape):
        det = []
        output = non_max_suppression(output_array, self.conf_thresh, 0.5)
        for i, det in enumerate(output):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords((self.short_size, self.short_size), det[:, :4], ori_shape).round()
        return det

class DetectProcessOld(ProcessBase):
    def __init__(self, short_size=640, conf_thresh=0.01):
        self.short_size = short_size
        self.conf_thresh = conf_thresh

    def preprocess(self, img):
        shape = img.shape  # orig hw
        img = letterbox(img, new_shape=(self.short_size, self.short_size))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = img.astype(np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        return img, shape

    def post_process(self, output_array, ori_shape):
        det = []
        # print(output_array, self.conf_thresh)
        output = non_max_suppression(output_array, self.conf_thresh, 0.5)
        # print(output)
        for i, det in enumerate(output):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords((self.short_size, self.short_size), det[:, :4], ori_shape).round()
        return det
    
class ClsProcess(ProcessBase):
    def __init__(self, short_size=384):
        pass

    def preprocess(self, image, input_size=416):
        img = cv2.resize(image, (input_size, input_size),interpolation=cv2.INTER_CUBIC)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = img / 255.0
        img -= mean
        img /= std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        input_data = np.float32(img)
        # print(input_data)
        return input_data

    def post_process(self, output_array):
        # print("output_array is:",output_array)
        x = softmax(output_array)
        # print("x is:",x)
        score = x[0, :].tolist()
        score_r = score.copy()
        score_r.sort(reverse=True)
        # print("score is:",score, 'score_r is:', score_r)
        index_list = []
        score_list = []
        for item in score_r[:5]:
            index_list.append(score.index(item))
            score_list.append(item)
        index = score.index(max(score))
        return [index_list, score_list]
        # x = softmax(output_array)
        # num_classes = x.size(1)
        # topk_probs, topk_indices = torch.topk(x, k=5)
        # index_list = topk_indices.tolist()[0] 
        # index_list_probs = topk_probs.tolist()[0]
        # return index_list, index_list_probs


class ClsProcess2(ProcessBase):
    def __init__(self, short_size=384):
        pass

    def preprocess(self, image, input_size=416):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV加载的图片通道顺序为BGR，需转为RGB
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        # img = cv2.copyMakeBorder(img, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0)
        input_data = img.astype(np.float32) / 255.0  # 标准化像素值到 [0, 1] 范围
        input_data = np.transpose(input_data, (2, 0, 1))  # 调整颜色通道顺序
        input_data = np.expand_dims(input_data, axis=0)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_data = normalize(torch.tensor(input_data)).numpy()
        return input_data

    def post_process(self, output_array):
        x = softmax(output_array)
        score = x[0, :].tolist()
        index = score.index(max(score))
        return index, score[index]
    

class ClsProcess3(ProcessBase):
    def __init__(self, short_size=672):
        pass

    def preprocess(self, image, input_size=672):
        img = cv2.resize(image, (input_size, input_size),interpolation=cv2.INTER_CUBIC)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        img = img / 255.0
        img -= mean
        img /= std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        input_data = np.float32(img)
        # print(input_data)
        return input_data

    def post_process(self, output_array):
        # print("output_array is:",output_array)
        x = softmax(output_array)
        # print("x is:",x)
        score = x[0, :].tolist()
        score_r = score.copy()
        score_r.sort(reverse=True)
        # print("score is:",score, 'score_r is:', score_r)
        index_list = []
        score_list = []
        for item in score_r[:5]:
            index_list.append(score.index(item))
            score_list.append(item)
        index = score.index(max(score))
        return [index_list, score_list]
        # x = softmax(output_array)
        # num_classes = x.size(1)
        # topk_probs, topk_indices = torch.topk(x, k=5)
        # index_list = topk_indices.tolist()[0] 
        # index_list_probs = topk_probs.tolist()[0]
        # return index_list, index_list_probs


class ValueProcess(ProcessBase):
    def __init__(self, short_size=224):
        pass

    def preprocess(self, image, input_size=224):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        (old_h, old_w, _) = image.shape
        new_w = int(input_size / old_h * old_w)
        new_h = input_size
        if old_w > old_h:
            new_w = input_size
            new_h = int(input_size / old_w * old_h)

        padding_h = input_size - new_h
        padding_w = input_size - new_w

        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = img / 255.0
        img -= mean
        img /= std
        img = cv2.copyMakeBorder(img, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0)
        img_ori = img.astype(np.float32)
        img = img_ori.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def post_process(self, output_array):
        x = softmax(output_array)
        score = x[0, :].tolist()
        index = score.index(max(score))
        return index, score[index]
