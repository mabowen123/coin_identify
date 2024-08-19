import os
import uuid

import cv2
import math
import time
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures.process import ProcessPoolExecutor

log_path = os.path.join(os.path.dirname(__file__), 'download.log')
handler = logging.FileHandler(log_path, encoding='UTF-8')
console = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(lineno)d | %(message)s')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger = logging.getLogger('download')
logger.setLevel(level=logging.DEBUG)
logger.addHandler(handler)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=False, help='data csv path')
parser.add_argument('--suffix', default='.jpg', help='save image suffix, option: [.png, .jpg, .jpeg]')
parser.add_argument('--thread', type=int, default=4, help='thread nums, defualt 4')
parser.add_argument('--process', type=int, default=4, help='process nums, default 4')
parser.add_argument('--url', default='', help='image download url')
parser.add_argument('--dst', default='detected_data', help='dst save path')
parser.add_argument('--per', type=int, default=1000, help='per process deal image nums')
parser.add_argument('--col', type=str, default='图片', help='csv column name (download url)')
parser.add_argument('--sample', type=int, help='sample')
args = parser.parse_args()

import torchvision.transforms as transforms
import onnxruntime
import os
from torch.backends import cudnn
import numpy as np
import torch
import torchvision
import requests
import time

weights_detect = '/Users/mabowen/Documents/www/coin_identify/classifyc/model/coindetect_0426.onnx'

cudnn.fastest = True
cudnn.benchmark = True
conf_thres = 0.01
iou_thres = 0.5
device = torch.device("cpu")

session_detect = onnxruntime.InferenceSession(weights_detect)
input_name_detect = session_detect.get_inputs()[0].name
output_name_detect = session_detect.get_outputs()[0].name


def pic_deal(image, input_size=224):
    img = letterbox(image, new_shape=(input_size, input_size))[0]
    img = transforms.ToTensor()(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img = normalize(img)
    img = img.numpy()
    img = np.expand_dims(img, axis=0)

    return img


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
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


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right xcd
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
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

    return output


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


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def detect(img0, save_path):
    """
    检测目标中钱币，如遇到多个目标，只返回检测分最大的值
    返回:
           numpy 格式图片
    """
    img = letterbox(img0, new_shape=(416, 416))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    prediction = session_detect.run([output_name_detect], {input_name_detect: img})
    pred = prediction[0]
    pred = torch.from_numpy(pred).to(device)

    # pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, merge=False, classes=None, agnostic=False)

    return_pics = []
    for i, det in enumerate(pred):
        # det = torch.from_numpy(det2)
        if det is not None and len(det):
            det[:, :4] = scale_coords((416, 416), det[:, :4], img0.shape).round()
            best_conf = 0
            new_list = sorted(det.numpy().tolist(), key=lambda x: x[4])
            # det = torch.Tensor(new_list[-1])
            num = 0
            for *xyxy, conf, cls in det:
                if conf > best_conf:
                    cut_img = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    # cut_img = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
                    # cut_img = cv2.cvtColor(cut_img, cv2.COLOR_RGB2BGR)
                    save_path_new = save_path + str(num) + '.jpg'
                    save_path_new = save_path_new.replace("zhuanhuayanse", "cut")
                    cut_img = circle_cut(cut_img)
                    cv2.imwrite(save_path_new, cut_img)
                    num += 1
                else:
                    continue
    return return_pics
    # pred2 = pred[0].numpy()
    # for i, det2 in enumerate(pred2):
    #     det = torch.from_numpy(det2)
    #     if det is not None and len(det):
    #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    #         best_conf = 0
    #         new_list = sorted(det.numpy().tolist(), key=lambda x: x[4])
    #         det = torch.Tensor(new_list[-1])
    #         # for *xyxy, conf, cls in det:
    #
    #         cut_img = img0[int(det[1]):int(det[3]), int(det[0]):int(det[2]), :]
    #         if det[4] > best_conf:
    #             save_path = save_path+str(i)+'.jpg'
    #             cv2.imwrite(save_path, cut_img)
    #         else:
    #             continue
    # return return_pics


def circle_cut(img_src):
    rows, cols, _ = img_src.shape
    img_mask = np.zeros((rows, cols, 3), np.uint8)
    img_mask[:, :, :] = 255
    img_mask = cv2.circle(img_mask, (int(cols / 2), int(rows / 2)), int(max(rows, cols) / 2 + 1), (0, 0, 0), -1)
    img_circle = cv2.add(img_src, img_mask)
    return img_circle


def thread_download(pi, ti, image, save_path):
    try:
        # logger.debug(f'start download {image}..., pi: {pi}, ti: {ti}')
        start_time = time.time()
        # save_path = os.path.join(save_path, uuid.uuid1().__str__() + args.suffix)
        # resp = requests.get(image)
        # img = np.frombuffer(resp.content, np.uint8)
        # image = 'data/第三套人民币伍角背面/ef2abbb0-697b-11ed-9f67-acde48001122.jpg'
        img = cv2.imread(image, cv2.IMREAD_ANYCOLOR)
        # cv2.imshow('Image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cut_pics = detect(img, save_path)
        # logger.debug(f'download {image} success, {pi}, ti: {ti}, time: {round(time.time() - start_time, 4)}s')
    except (Exception,) as e:
        logger.error(f'del {image} error, pi: {pi}, ti{ti}, error: {e}', exc_info=True)


def make_dirs(path_list):
    for p in path_list:
        if os.path.exists(p):
            continue
        else:
            os.makedirs(p)


if __name__ == '__main__':
    ori_path = '/Users/mabowen/Downloads/zhuanhuayanse'
    g = os.walk(ori_path)
    for path, dir_list, file_list in g:
        image_list = []
        for file_name in file_list:
            print(file_name)
            if file_name == '.DS_Store':
                continue
            name = os.path.join(path, file_name)
            image_list.append(name)
            save_path = os.path.join(args.dst, path)
            make_dirs([save_path])
            save_name = os.path.join(save_path, file_name)
            thread_download(1, 1, name, save_name)
    logger.info(f'{ori_path} download successful files')
