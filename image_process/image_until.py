import json
import os.path
import random
import sys
import time
import cv2
import requests
import numpy as np
import urllib3

sys.path.append("../")

from tool.print import print_with_timestamp

max_retry = 5


def sleep():
    time.sleep(random.randint(1, 5))


def save_image(image_url, version, save_path):
    start_time = time.time()
    ti = random.randint(1, 10000000)
    img_id = str(int(start_time * 10) + ti)
    save_path = os.path.join(save_path, f"{version}_{img_id}.jpg")
    for index in range(max_retry):
        headers = {'Connection': 'close'}
        r = requests.get(image_url, timeout=60, headers=headers)
        r.close()
        if r.status_code == 200:
            with open(
                    save_path,
                    "wb",
            ) as f:
                f.write(r.content)
            break
        sleep()

    return save_path


# 图片灰度处理
def image_gray(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(image_path, img)


def circle_cut(img_src):
    rows, cols, _ = img_src.shape
    img_mask = np.zeros((rows, cols, 3), np.uint8)
    img_mask[:, :, :] = 255
    # 用min
    img_mask = cv2.circle(img_mask, (int(cols / 2), int(rows / 2)), int(min(rows, cols) / 2 - 2), (0, 0, 0), -1)
    img_circle = cv2.add(img_src, img_mask)
    return img_circle


def cut_img_code(img_name):
    origin_img = cv2.imread(img_name)
    ff = open(img_name, "rb")
    files = {"file": ff}
    try:
        for i in range(max_retry):
            headers = {'Connection': 'close'}
            res = requests.post(
                "http://imgt.wpt.la/coin-feature/api/inference", files=files, timeout=60, headers=headers
            )
            res.close()
            # 返回值示例
            # {"code":0,"data":[],"data_v2":[],"nowTime":1709199710,"others":{"box":[3,4,641,637],"detect_label":0,"detect_score":0.9253292083740234}}
            if res.status_code == 200:
                res_js = json.loads(res.content)
                point = res_js["others"]["box"]
                main_img = origin_img[point[1]: point[3], point[0]: point[2]]
                img_circle = circle_cut(main_img)
                return img_circle
            sleep()
    except (Exception,) as e:
        print_with_timestamp("cut_img_code处理出错 img_name:%s err: %s" % (img_name, e))
        return ""


def scale_img(right_image):
    try:
        right_image = cut_img_code(right_image)
        if right_image == "":
            return ["", False]
        right_image_szie = right_image.shape
        right_image_szie_h = right_image_szie[0]
        right_image_szie_w = right_image_szie[1]
        scale = np.sqrt(640 * 640 / (right_image_szie_w * right_image_szie_h))
        width_new = np.intp(right_image_szie_w * scale)
        height_new = np.intp(right_image_szie_h * scale)
        right_image_scaled = cv2.resize(right_image, (width_new, height_new))
        return [right_image_scaled, True]
    except (Exception,) as e:
        print_with_timestamp("scale_img处理出错 right_iamge:%s err: %s" % (right_image, e))
        return ["", False]
