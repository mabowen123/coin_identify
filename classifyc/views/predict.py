# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:38:41 2018

@author: zty
"""
import torch
import numpy as np
import config
from config import replace_dict, use_trt_flag, DETECT_TRT_PATH, DETECT_MODEL_PATH,DETECT_MODEL_OLD_PATH, \
    CLASSIFY_TRT_PATH, CLASSIFY_MODEL_PATH, COIN_VALUE_MODEL_PATH, COIN_VALUE_TRT_PATH, print_info,old_coin_name_dicts,CLASSIFY_OLD_MODEL_PATH,CLASSIFY_RESNET_MODEL_PATH,CLASSIFY_OLD_MODEL_PATH2, CLASSIFY_OLD_MODEL_PATH_1129,DETECT_MODEL_BEI_PATH 
from config import class_names, zhuzaoju_list, coin_value_names,angjiang_class_names,jingtui_class_names,fuyang_class_names,all_class_names,all_class_names_42,all_class_names_12,all_class_names_9,all_class_names_7,all_class_names_4,all_class_names_5,all_class_names_16,all_class_names_35,all_class_names_43,all_class_names_15,all_class_names_17,all_class_names_82,all_class_names_27
import cv2
from torch.backends import cudnn
import time

replace_keys = replace_dict.keys()
device_type = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device_type)
device = torch.device(device_type)

if use_trt_flag:
    from views.predict_trt import DetectTrt as Detect, ValueTrt as Value, ClsTrt as Cls
    # from views.predict_impl import ClsOnnx as Cls
    detect_session = Detect(DETECT_TRT_PATH, 416, conf_thresh=config.conf_thres)
    detect_session_old = Detect(DETECT_TRT_PATH, 416, conf_thresh=config.conf_thres)
    detect_session_bei = Detect(DETECT_TRT_PATH, 416, conf_thresh=config.conf_thres)
    cls_session = Cls(CLASSIFY_TRT_PATH)
    # cls_session = Cls(CLASSIFY_MODEL_PATH)
    value_session = Value(COIN_VALUE_TRT_PATH)
else:
    from views.predict_impl import DetectOnnx as Detect, ClsOnnx as Cls, ClsOnnx3 as Cls3, ValueOnnx as Value, DetectOnnx2 as Detect2

    detect_session = Detect(DETECT_MODEL_PATH, 416, conf_thresh=config.conf_thres)
    detect_session_old = Detect2(DETECT_MODEL_OLD_PATH, 640, conf_thresh=config.conf_thres_old)
    detect_session_bei = Detect2(DETECT_MODEL_BEI_PATH, 640, conf_thresh=config.conf_thres_old)
    # detect_session_bei = Detect(DETECT_MODEL_BEI_PATH, 416, conf_thresh=config.conf_thres_old)
    
    # b8的模型
    if 'efficientnet_b8' in CLASSIFY_OLD_MODEL_PATH_1129:
        print("当前启动的模型是b8")
        cls_session = Cls3(CLASSIFY_MODEL_PATH)
        cls_old_session = Cls3(CLASSIFY_OLD_MODEL_PATH)
        cls_old_session2 = Cls3(CLASSIFY_OLD_MODEL_PATH_1129)
    else: # 默认是b5
        cls_session = Cls(CLASSIFY_MODEL_PATH)
        cls_old_session = Cls(CLASSIFY_OLD_MODEL_PATH)
        cls_old_session2 = Cls(CLASSIFY_OLD_MODEL_PATH_1129)
    print("CLASSIFY_OLD_MODEL_PATH_1129", CLASSIFY_OLD_MODEL_PATH_1129)
    print("DETECT_MODEL_BEI_PATH", DETECT_MODEL_BEI_PATH)
    value_session = Value(COIN_VALUE_MODEL_PATH)

cudnn.fastest = True
cudnn.benchmark = True


def pic_deal(image, input_size=224):
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


def classify_back_coin(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img = pic_deal(image)
    cls_index, cls_score = cls_session(image)
    rec_classify = []
    for coin_item in cls_index:
        rec_classify.append(class_names[coin_item]) 
    return rec_classify, cls_score


def circle_cut(img_src):
    rows, cols, _ = img_src.shape
    img_mask = np.zeros((rows, cols, 3), np.uint8)
    img_mask[:, :, :] = 255
    # 用min
    img_mask = cv2.circle(img_mask, (int(cols / 2), int(rows / 2)), int(min(rows, cols) / 2 - 2), (0, 0, 0), -1)
    # # # 用max
    # img_mask = cv2.circle(img_mask, (int(cols / 2), int(rows / 2)), int(max(rows, cols) / 2 + 1), (0, 0, 0), -1)
    
    img_circle = cv2.add(img_src, img_mask)
    return img_circle


def merge_two_cur_img(img_circle_left, img_circle_right):
    if img_circle_left is None or img_circle_right is None:
        return None
    resized_image2 = cv2.resize(img_circle_right, (img_circle_left.shape[1], img_circle_left.shape[0]))
    new_img = cv2.hconcat([img_circle_left, resized_image2])
    return new_img


def classifyc_coin(cls_crop_img):
    print("begin classifyc_coin")
    rec_classify, cls_score = all_class_names(cls_crop_img)
    # print("adfer cls_label:", rec_classify)
    # return [replace_dict[coin_item] if coin_item in replace_keys for coin_item in coin], [], cls_score
    # if coin in replace_keys:
        
    return rec_classify, [], cls_score


def predict_two_pics(src_img_1, src_img_2):
    cls_crop_img_1 = detect_coin(src_img_1)
    cls_crop_img_2 = detect_coin(src_img_2)
    if cls_crop_img_1 is None or cls_crop_img_2 is None:
        return "未检测到钱币", [], 0
    merge_two_img = merge_two_cur_img(cls_crop_img_1, cls_crop_img_2)
    # cv2.imwrite('/gpu1-data/datasets/coin_class/实物图/merge_process_fix_cut_two/5C or 10C Capped Bust/tmp1.jpg', merge_two_img)
    coin, [], cls_score = classifyc_coin(merge_two_img)
    return coin, [], cls_score


def predict_old_feature(src_img):
    coin, [], cls_score = detect_coin_old_1129(src_img)
    return coin, [], cls_score
    

def detect_coin(src_img):
    detect_objects = detect_session(src_img)
    if detect_objects is not None and len(detect_objects):
        *xyxy, det_score, det_cls = detect_objects[0]
        cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
        cls_crop_img = circle_cut(cls_crop_img)
        return cls_crop_img
        # coin, cls_score = classify_back_coin(cls_crop_img)
        # if coin in replace_keys:
        #     return replace_dict[coin], [int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])], cls_score
        # return coin, [int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])], cls_score
    else:
        # return "未检测到钱币", [], 0
        return None

    

def classify_back_old_coin(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img = pic_deal(image)
    cls_index, cls_score = cls_old_session(image)
    print(cls_index, cls_score)
    rec_classify = []
    for coin_item in cls_index:
        rec_classify.append(all_class_names_82[coin_item]) 
    return rec_classify, cls_score

def classify_back_old_coin2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img = pic_deal(image)
    cls_index, cls_score = cls_old_session2(image)
    print(cls_index, cls_score)
    rec_classify = []
    for coin_item in cls_index:
        rec_classify.append(all_class_names_27[coin_item]) 
    return rec_classify, cls_score


def classifyc_old_coin(cls_crop_img):
    print("begin classifyc_coin")
    rec_classify, cls_score = classify_back_old_coin(cls_crop_img)
    # print("adfer cls_label:", rec_classify)
    # return [replace_dict[coin_item] if coin_item in replace_keys for coin_item in coin], [], cls_score
    # if coin in replace_keys:
        
    return rec_classify, [], cls_score

def classifyc_old_coin2(cls_crop_img):
    print("begin classifyc_coin")
    rec_classify, cls_score = classify_back_old_coin2(cls_crop_img)
    # print("adfer cls_label:", rec_classify)
    # return [replace_dict[coin_item] if coin_item in replace_keys for coin_item in coin], [], cls_score
    # if coin in replace_keys:
        
    return rec_classify, [], cls_score
    
# def detect_coin_old(src_img):
#     detect_objects = detect_session_old(src_img)
#     rec_classify_all=[]
#     cls_score_all =[]
#     if detect_objects is not None and len(detect_objects):
#         cls_list = []
#         for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#             cls_list.append(old_coin_name_dicts[int(det_cls.numpy().tolist())])
#         for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#             label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
#             print(xyxy,det_cls,old_coin_name_dicts, label)
#             cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
#             cls_crop_img = circle_cut(cls_crop_img)
#             if old_coin_name_dicts[int(det_cls.numpy().tolist())] == '穿':
#                 continue
#             rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # if '重' in cls_list:
#             # if '重' in cls_list and (label == '崇' or label == '宁'):
#             #     rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # elif '通' in cls_list and (label == '崇' or label == '通'):
#             #     rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # if '重' in cls_list and (label == '重' or label == '宝'):
#             #     rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # elif '通' in cls_list and (label == '宁' or label == '宝'):
#             #     rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # else:
#             #     continue
#             rec_classify = [ item +label for item in rec_classify]
#             rec_classify_all+=rec_classify
#             cls_score_all+=cls_score
#             print(rec_classify_all, [], cls_score_all)
#         return rec_classify_all, [], cls_score_all
#     else:
#         # return "未检测到钱币", [], 0
#         return None
 
def merge_two_cur_img2(img1, img2):
    if img1 =='' or img2 =='':
        return ''
    # print('img1 and img2 is not null')
    img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    # print('img1 and img2 rotate is not null')
    height1, width1,_ = img1.shape
    height2, width2,_ = img2.shape

    max_width = max(width1, width2)
    max_height = max(height1, height2)

    img1_resized = cv2.resize(img1, (max_width, max_height))
    img2_resized = cv2.resize(img2, (max_width, max_height))
    print('img1 and img2 size is: ', height1, width1, height2, width2)
    new_img = cv2.vconcat([img1_resized, img2_resized])
    return new_img

# def detect_coin_old(src_img):
#     print("begin detect_coin_old")
#     detect_objects = detect_session_old(src_img)
#     print("detect_objects size is,", len(detect_objects))
#     if detect_objects is not None and len(detect_objects):
#         print("begin detect_coin_old is, detect_objects not null")
#         img_map = {}
#         for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#             if index == 2:
#                 continue
#             print(index)
#             label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
#             print(xyxy,det_cls,old_coin_name_dicts, label)
#             cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
#             # cls_crop_img = circle_cut(cls_crop_img)
#             if label in ['宁宝','重宝']:
#                 img_map[0] = cls_crop_img
#             else:
#                 img_map[1] = cls_crop_img
#         print("merge img begin")
#         new_img = merge_two_cur_img2(img_map[0], img_map[1])
#         # new_img = img_map[0]
#         print("merge img done")
#         rec_classify, [], cls_score = classifyc_old_coin(new_img)
#         print(rec_classify, cls_score)
#         return rec_classify, [], cls_score
#     else:
#         # return "未检测到钱币", [], 0
#         return None


# def detect_coin_old(src_img):
#     detect_objects = detect_session_old(src_img)
#     rec_classify_all=[]
#     cls_score_all =[]
#     print(detect_objects)
#     # path = "/gpu1-data/datasets/old_coin_class/ytst_data/单个字/崇宁通宝/宁/"
#     if detect_objects is not None and len(detect_objects):
#         cls_list = []
#         for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#             cls_list.append(old_coin_name_dicts[int(det_cls.numpy().tolist())])
#         for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#             label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
#             print(xyxy,det_cls,old_coin_name_dicts, label)
#             cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
#             # cls_crop_img = circle_cut(cls_crop_img)
#             label = old_coin_name_dicts[int(det_cls.numpy().tolist())]
#             if '崇' in label:
#                 cls_crop_img = cv2.rotate(cls_crop_img, cv2.ROTATE_90_CLOCKWISE)
#             if label == '穿':
#                 continue
#             rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # if '重' in cls_list:
#             # if '重' in cls_list and (label == '崇' or label == '宁'):
#             #     rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # elif '通' in cls_list and (label == '崇' or label == '通'):
#             #     rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # if '重' in cls_list and (label == '重' or label == '宝'):
#             #     rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # elif '通' in cls_list and (label == '宁' or label == '宝'):
#             #     rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # else:
#             #     continue
#             # rec_classify = [ item for item in rec_classify]
#             rec_classify_all+=rec_classify[:1]
#             cls_score_all+=cls_score[:1]
#             print(rec_classify_all, [], cls_score_all)
#             # start_time = time.time()
#             # id = int(start_time * 10)
#             # print(id, path+str(id)+'.jpg')
#             # cv2.imwrite(path+str(id)+'_'+rec_classify_all[0] + '.jpg', cls_crop_img)
#         return rec_classify_all, [], cls_score_all
#     else:
#         # return "未检测到钱币", [], 0
#         return None


# def detect_coin_old(src_img):
#     detect_objects = detect_session(src_img)
#     rec_classify_all=[]
#     cls_score_all =[]
#     print(detect_objects)
#     # path = "/gpu1-data/datasets/old_coin_class/ytst_data/单个字/崇宁通宝/宁/"
#     if detect_objects is not None and len(detect_objects):
#         cls_list = []
#         for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#             # label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
#             print(xyxy,det_cls)
#             cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
#             cls_crop_img = circle_cut(cls_crop_img)
#             rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             rec_classify_all+=rec_classify[:1]
#             cls_score_all+=cls_score[:1]
#             print(rec_classify_all, [], cls_score_all)
#             # start_time = time.time()
#             # id = int(start_time * 10)
#             # print(id, path+str(id)+'.jpg')
#             # cv2.imwrite(path+str(id)+'_'+rec_classify_all[0] + '.jpg', cls_crop_img)
#         return rec_classify_all, [], cls_score_all
#     else:
#         # return "未检测到钱币", [], 0
#         return None

# def detect_coin_old(src_img):
#     detect_objects = detect_session_old(src_img)
#     rec_classify_all=[]
#     cls_score_all =[]
#     print(detect_objects)
#     path = "/gpu1-data/datasets/old_coin_class2/one_feature/cut_gray_one/"
#     if detect_objects is not None and len(detect_objects):
#         cls_list = []
#         for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#             label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
#             print(xyxy,det_cls)
#             if label != '重':
#                 continue
#             cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
#             # cls_crop_img = circle_cut(cls_crop_img)
#             # rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#             # rec_classify_all+=rec_classify[:1]
#             # cls_score_all+=cls_score[:1]
#             # print(rec_classify_all, [], cls_score_all)
#             # start_time = time.time()
#             # id = int(start_time * 10)
#             # print(id, path+str(id)+'.jpg')
#             cls_crop_img = scale_img(cls_crop_img)
#             cv2.imwrite(path + 'tmp.jpg', cls_crop_img)
#         return rec_classify_all, [], cls_score_all
#     else:
#         # return "未检测到钱币", [], 0
#         return None

def scale_img(rightIamge):
    # rightIamge = cut_img_code(rightIamgeName)
    rightIamge_szie = rightIamge.shape
    rightIamge_szie_h = rightIamge_szie[0]
    rightIamge_szie_w = rightIamge_szie[1]
    scale = np.sqrt(640 * 640 / (rightIamge_szie_w * rightIamge_szie_h))
    width_new = np.int0(rightIamge_szie_w * scale)
    height_new = np.int0(rightIamge_szie_w * scale)
    right_image_scaled = cv2.resize(rightIamge, (width_new, height_new))
    return right_image_scaled

# def detect_coin_old(src_img):
#     rec_classify_all=[]
#     cls_score_all =[]
#     #大分类   
#     path = "/gpu1-data/datasets/old_coin_class2/one_feature_4/test/"
#     detect_objects = detect_session(src_img) 
#     if detect_objects is not None and len(detect_objects):
#         for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#             label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
#             # print(index, xyxy)
#             if index >0:
#                 continue
#             cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
#             cls_crop_img = circle_cut(cls_crop_img)
#             cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_BGR2GRAY)
#             cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_GRAY2RGB)
#             cls_crop_img = scale_img(cls_crop_img)
#             # cv2.imwrite(path + 'tmp.jpg', cls_crop_img)
#             rec_classify, [], cls_score = classifyc_old_coin2(cls_crop_img)
#             rec_classify_all+=rec_classify[:3]
#             cls_score_all+=cls_score[:3]
    
#     print(cls_score_all, cls_score_all)
    
#     # detect_objects = detect_session_old(src_img)
#     # if detect_objects is not None and len(detect_objects):
#     #     name_list = []
#     #     for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#     #         label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
#     #         # print(label, xyxy)
#     #         if label == '穿':
#     #             continue
#     #         if label in name_list:
#     #             continue
#     #         name_list.append(label)
#     #         cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
#     #         cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_BGR2GRAY)
#     #         cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_GRAY2RGB)
#     #         cls_crop_img = scale_img(cls_crop_img)
#     #         rec_classify, [], cls_score = classifyc_old_coin(cls_crop_img)
#     #         rec_classify_all+=rec_classify[:2]
#     #         cls_score_all+=cls_score[:2]
            
#     # print(rec_classify_all, cls_score_all)

#     return rec_classify_all, [], cls_score_all

import requests
import json

def ytst_res(fullname, label1, name):
    ff = open(fullname, 'rb')
    files = {'file': ff}
    post_js = {
            "label1": label1
        }
    print(label1, name)
    for index in range(3):
        res = requests.post('http://127.0.0.1:85/coin-agent/api/searchfile', data=post_js, files=files, timeout=60)
        # print(res)
        if res.status_code == 200:
            res_js = json.loads(res.content)
            if res_js['code'] == 0:
                labels = dict()
                labels_num = dict()
                predict_label_y = res_js['data']
                print(predict_label_y)
                for i in range(len(predict_label_y)):
                    item = predict_label_y[i]['2nd_label']
                    if name not in item:
                        continue
                    if item in labels:
                        labels[item] += predict_label_y[i]['distance']
                        labels_num[item] += 1
                    else:
                        labels[item] = predict_label_y[i]['distance']
                        labels_num[item] = 1
                for item in labels:
                    labels[item] = labels[item]/labels_num[item]
                print(labels)
                if len(labels) ==0:
                    return '正' + name, 0
                sorted_list_by_key = sorted(labels.items(), key=lambda x: x[1])
                predict_label = sorted_list_by_key[0][0]
    return predict_label,sorted_list_by_key[0][1]
        

def detect_coin_old(src_img):
    rec_classify_all=[]
    cls_score_all =[]
    #大分类   
    path = "/gpu1-data/datasets/old_coin_class2/one_feature_4/test/"
    detect_objects = detect_session(src_img) 
    if detect_objects is not None and len(detect_objects):
        for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
            label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
            # print(index, xyxy)
            if index >0:
                continue
            cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            cls_crop_img = circle_cut(cls_crop_img)
            cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_BGR2GRAY)
            cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_GRAY2RGB)
            cls_crop_img = scale_img(cls_crop_img)
            # cv2.imwrite(path + 'tmp.jpg', cls_crop_img)
            rec_classify, [], cls_score = classifyc_old_coin2(cls_crop_img)
            rec_classify_all+=rec_classify[:1]
            cls_score_all+=cls_score[:1]
    
    print(cls_score_all, cls_score_all, rec_classify_all[0][:-1])
    
    detect_objects = detect_session_old(src_img)
    if detect_objects is not None and len(detect_objects):
        name_list = []
        for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
            label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
            print(label, xyxy)
            if label == '穿':
                continue
            if label in name_list:
                continue
            name_list.append(label)
            cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_BGR2GRAY)
            cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_GRAY2RGB)
            cls_crop_img = scale_img(cls_crop_img)
            cv2.imwrite(path + 'tmp.jpg', cls_crop_img)
            predict_label,cls_score = ytst_res(path + 'tmp.jpg',rec_classify_all[0][:-1], label)
            rec_classify_all.append(predict_label)
            cls_score_all.append(cls_score)
            
    print(rec_classify_all, cls_score_all)

    return rec_classify_all, [], cls_score_all




def detect_coin_old_1129(src_img):
    rec_classify_all=[]
    cls_score_all =[]
    #大分类   
    detect_objects = detect_session(src_img) 
    if detect_objects is not None and len(detect_objects):
        for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
            label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
            # print(index, xyxy)
            if index >0:
                continue
            # cls_crop_img = src_img
            cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            cls_crop_img = circle_cut(cls_crop_img)
            cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_BGR2GRAY)
            cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_GRAY2RGB)
            cls_crop_img = scale_img(cls_crop_img)
            rec_classify, [], cls_score = classifyc_old_coin2(cls_crop_img)
            rec_classify_all+=rec_classify[:5]
            cls_score = [round(it,3) for it in cls_score]
            cls_score_all+=cls_score[:5]
    
    # print(cls_score_all, cls_score_all, rec_classify_all[0][:-1])

    return rec_classify_all, [], cls_score_all

def cut_pic_half(iamge_pic):
    iamge_pic_szie = iamge_pic.shape
    iamge_pic_szie_h = iamge_pic_szie[0]
    iamge_pic_szie_w = iamge_pic_szie[1]
    len_x_1 = int(iamge_pic_szie_h / 2)
    len_y_1 = int(iamge_pic_szie_w / 2)
    imgs_up = iamge_pic[:len_x_1,:]
    imgs_down = iamge_pic[len_x_1:,:]
    imgs_left = iamge_pic[:,:len_y_1]
    imgs_right = iamge_pic[:,len_y_1:]
    return imgs_up, imgs_down, imgs_left, imgs_right
 
def panbie_label(predict_label_c, predict_score):
    label = ['星', '月', '直纹']
    wenli_label_dict = {}
    for i in range(len(predict_label_c)):
        item = predict_label_c[i]
        score = predict_score[i]
        if item in label:
            wenli_label_dict[item,i] = score
    if len(wenli_label_dict) == 0:
        return '光背', predict_score[0]
    res = sorted(wenli_label_dict.items(), key=lambda d: d[1])[0]
    if res[0][1] == 0:
        return '背上' + res[0][0], res[1]
    if res[0][1] == 1:
        return '背下' + res[0][0], res[1]
    if res[0][1] == 2:
        return '背左' + res[0][0], res[1]
    if res[0][1] == 3:
        return '背右' + res[0][0], res[1]
    

# def detect_coin_old_1129(src_img):
#     rec_classify_all=[]
#     cls_score_all =[]
#     rec_all=[]
#     score_all =[]
#     #大分类   
#     detect_objects = detect_session(src_img) 
#     if detect_objects is not None and len(detect_objects):
#         for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
#             label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
#             # print(index, xyxy)
#             if index >0:
#                 continue
#             cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
#             cls_crop_img = circle_cut(cls_crop_img)
#             cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_BGR2GRAY)
#             cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_GRAY2RGB)
#             cls_crop_img = scale_img(cls_crop_img)
#             imgs_up, imgs_down, imgs_left, imgs_right = cut_pic_half(cls_crop_img)
            
#             rec_classify_up, [], cls_score_up = classifyc_old_coin2(imgs_up)
#             rec_classify_all+=rec_classify_up[:1]
#             cls_score = [round(it,3) for it in cls_score_up]
#             cls_score_all+=cls_score[:1]
            
#             rec_classify_down, [], cls_score_down = classifyc_old_coin2(imgs_down)
#             rec_classify_all+=rec_classify_down[:1]
#             cls_score = [round(it,3) for it in cls_score_down]
#             cls_score_all+=cls_score[:1]
            
#             rec_classify_left, [], cls_score_left = classifyc_old_coin2(imgs_left)
#             rec_classify_all+=rec_classify_left[:1]
#             cls_score = [round(it,3) for it in cls_score_left]
#             cls_score_all+=cls_score[:1]
            
#             rec_classify_right, [], cls_score_right = classifyc_old_coin2(imgs_right)
#             rec_classify_all+=rec_classify_right[:1]
#             cls_score = [round(it,3) for it in cls_score_right]
#             cls_score_all+=cls_score[:1]
    
#     # print(cls_score_all, cls_score_all, rec_classify_all[0][:-1])
#     res_label, res_score = panbie_label(rec_classify_all, cls_score_all)
#     rec_all.append(res_label)
#     score_all.append(res_score)
#     return rec_all, [], score_all



def feature_point(label_name, xyxy_1, xyxy_0):
    if xyxy_1[0] - xyxy_0[0] > 0 :
        return '背下' + label_name
    if xyxy_1[0] - xyxy_0[0] < 0 :
        return '背上' + label_name
    if xyxy_1[1] - xyxy_0[1] > 0 :
        return '背右' + label_name
    if xyxy_1[1] - xyxy_0[1] < 0 :
        return '背左' + label_name
            

def detect_coin_old_bei(src_img):
    rec_classify_all=[]
    cls_score_all =[]
    #大分类   
    detect_objects = detect_session_bei(src_img) 
    point_dict = {}
    if detect_objects is not None and len(detect_objects):
        for index, (*xyxy, det_score, det_cls) in enumerate(detect_objects):
            label =  old_coin_name_dicts[int(det_cls.numpy().tolist())]
            print(index, label, xyxy,det_cls.numpy().tolist())
            cls_crop_img = src_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
            point_dict[label] = xyxy
            
    if len(point_dict) == 1 or '穿口' not in point_dict:
        rec_classify_all = ['光背']
        cls_score_all = [1.0]
    else:
        xyxy_0 = point_dict['穿口']
        rec_classify_all = ['光背']
        cls_score_all = [1.0]
        if '月' in point_dict:
            xyxy_1 = point_dict['月']
            label_name = feature_point('月', xyxy_1, xyxy_0)
            rec_classify_all = [label_name]
            cls_score_all = [1.0]
        elif '星' in point_dict:
            xyxy_1 = point_dict['星']
            label_name = feature_point('星', xyxy_1, xyxy_0)
            rec_classify_all = [label_name]
            cls_score_all = [1.0]
        elif '直纹' in point_dict:
            xyxy_1 = point_dict['直纹']
            label_name = feature_point('直纹', xyxy_1, xyxy_0)
            rec_classify_all = [label_name]
            cls_score_all = [1.0]
            
            # cv2.imwrite('/gpu1-data/datasets/1129_old_coin_class/origin/load_pic_gray_cut_背面/测试/tmp_' + str(index) + '.jpg', cls_crop_img)
            # cv2.imwrite('./ytst_res_error_宝2/左实右预_实际_'+ dir2+'_预测_' + predict_label + '_'+filename +'.jpg',img)
            
#             cls_crop_img = circle_cut(cls_crop_img)
#             cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_BGR2GRAY)
#             cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_GRAY2RGB)
#             cls_crop_img = scale_img(cls_crop_img)
#             rec_classify, [], cls_score = classifyc_old_coin2(cls_crop_img)
#             rec_classify_all+=rec_classify[:5]
#             cls_score = [round(it,3) for it in cls_score]
#             cls_score_all+=cls_score[:5]
    
    # print(cls_score_all, cls_score_all, rec_classify_all[0][:-1])

    return rec_classify_all, [], cls_score_all

# def detect_coin_old_1129(src_img):
#     rec_classify_all=[]
#     cls_score_all =[]
#     #大分类  
#     cls_crop_img = src_img
#     # cls_crop_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
#     # cls_crop_img = cv2.cvtColor(cls_crop_img, cv2.COLOR_GRAY2RGB)
#     rec_classify, [], cls_score = classifyc_old_coin2(cls_crop_img)
#     rec_classify_all+=rec_classify[:5]
#     cls_score = [round(it,3) for it in cls_score]
#     cls_score_all+=cls_score[:5]
    
#     # print(cls_score_all, cls_score_all, rec_classify_all[0][:-1])

#     return rec_classify_all, [], cls_score_all