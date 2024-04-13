# -*- coding:utf-8 -*-
# @Time : 2020/6/23 10:44 AM
# @Author: shulin sun
# @File : class_onnx.py
import cv2
import numpy as np
from torchvision import transforms

from classifyc.config import class_names, zhuzaoju_list, coin_value_names
from classifyc.views.common import session_back, session_value
# from classifyc.views.predict_impl import post_process_classify


def pic_deal(image, input_size=224):
    (old_h, old_w, _) = image.shape
    new_w = int(input_size / old_h * old_w)
    new_h = input_size
    if old_w > old_h:
        new_w = input_size
        new_h = int(input_size / old_w * old_h)

    padding_h = input_size - new_h
    padding_w = input_size - new_w

    img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img = transforms.ToTensor()(img)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img = normalize(img)
    img = img.numpy()
    img = img.transpose(1, 2, 0)

    img = cv2.copyMakeBorder(img, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=0)

    img_ori = img.astype(np.float32)

    img = img_ori.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)

    return img


# def classify_back_coin(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = pic_deal(image)
#     output = session_back.predict(img)
#     index = post_process_classify(output)
#     rec_classify = class_names[index]
#     if rec_classify in zhuzaoju_list:
#         output_back = session_value.predict(img)
#         index_back = post_process_classify(output_back)
#         coin_value = coin_value_names[index_back]
#         if coin_value == "无面值":
#             return rec_classify
#         else:
#             return rec_classify + coin_value_names[index_back]
#     else:
#         if rec_classify in ['非钱币', '开集', '无背景或垂直', '其他纸币', '贰角银币', '香港钱币', '壹毫镍币']:
#             return '其他'
#         return rec_classify
