import json
import random
import string
import time

import cv2
import numpy as np
import pandas as pd
import requests

user_agent = "DouquanPro/6.2.4 (iPhone; iOS 17.4.1; Scale/3.00)"
Accept_Language = "zh-Hans-CN;q=1, en-CN;q=0.9, zh-Hant-HK;q=0.8, io-CN;q=0.7"
csv_path = "/Users/mabowen/Downloads/钱币识别-04北宋-淳化元宝-03W4.csv"
max_retry = 5
dq_token = ""


def up_dq_img(file_content, pic_img):
    for i in range(max_retry):
        # 发送POST请求
        r = requests.post("https://api.douquan.com/api/file/upload",
                          headers={
                              "Host": "api.douquan.com",
                              "User-Agent": user_agent,
                              "Accept-Language": Accept_Language,
                          },
                          files={
                              'file': (
                                  "{}{}".format(int(time.time()), ".jpg"), file_content, 'image/jpg/png/jpeg')},
                          data={"fileType": 1, 'file': ""})
        r.close()
        sleep()
        if r.status_code == 200:
            print(f"{pic_img}成功上传斗泉{r.json()}")
            return r.json()['data']['url']
    return ""


def match_query(front_dq_pic, back_img_dq_pic):
    r = requests.get("https://api.douquan.com/api/coinKind/versionMatchingQuery", params={
        "backImage": back_img_dq_pic,
        "frontImage": front_dq_pic,
        "imageSource": "1",
        "visibleStatus": "0"
    }, headers={
        "Host": "api.douquan.com",
        "universalId": ''.join(random.choices(string.ascii_letters + string.digits, k=28)),
        "deviceType": "2",
        "clientType": "dqAppIos",
        "Accept": "*/*",
        "User-Agent": user_agent,
        "Accept-Language": Accept_Language,
        "token": dq_token,
        "appVersion": "6.2.4"
    })
    r.close()
    sleep()
    print(f"斗泉对版结果{r.json()}")
    return r.json()['data']


def sleep():
    time.sleep(random.uniform(0.2, 2))


def get_img(pic_url):
    for i in range(max_retry):
        headers = {'Connection': 'close'}
        r = requests.get(pic_url, timeout=500, headers=headers)
        r.close()
        if r.status_code == 200:
            return r.content

    return ""


def cut_img(pic_url):
    img_content = get_img(pic_url)
    if img_content == "":
        print(f"读取{pic_url}失败")
        return ""
    print(f"读取{pic_url}成功")
    headers = {'Connection': 'close'}
    for i in range(max_retry):
        r = requests.post(
            "http://imgt.wpt.la/coin-feature/api/inference", files={"file": img_content}, timeout=300, headers=headers
        )
        r.close()
        if r.status_code == 200:
            point = json.loads(r.content)["others"]["box"]
            print(f"抠图{pic_url}成功")
            origin_img = cv2.imdecode(np.frombuffer(img_content, dtype=np.uint8), cv2.IMREAD_COLOR)
            main_img = origin_img[point[1]: point[3], point[0]: point[2]]
            rows, cols, _ = main_img.shape
            img_mask = np.zeros((rows, cols, 3), np.uint8)
            img_mask[:, :, :] = 255
            img_mask = cv2.circle(img_mask, (int(cols / 2), int(rows / 2)), int(max(rows, cols) / 2 + 1), (0, 0, 0),
                                  -1)
            img_circle = cv2.add(main_img, img_mask)
            right_image_size_h = img_circle.shape[0]
            right_image_size_w = img_circle.shape[1]
            scale = np.sqrt(640 * 640 / (right_image_size_h * right_image_size_w))
            width_new = np.intp(right_image_size_h * scale)
            height_new = np.intp(right_image_size_w * scale)
            _, image_array = cv2.imencode('.jpg', cv2.resize(img_circle, (width_new, height_new)))
            return image_array.tobytes()
    return ""


df = pd.read_csv(csv_path)
df_cleaned = df.dropna(subset=['原图链接'])
total = len(df_cleaned)
success = error = 0
for index, item in df_cleaned.iterrows():
    if len(item) > 0:
        front_pic, back_img = item["原图链接"].split(",")
        try:
            front_cut_pic = cut_img(front_pic)
            back_cut_pic = cut_img(back_img)
            if front_cut_pic != "" or back_cut_pic != "":
                res = match_query(up_dq_img(front_cut_pic, front_pic), up_dq_img(back_cut_pic, back_img))
                if res['coinKindItemVersionInfo'] is not None and "name" in res["coinKindItemVersionInfo"]:
                    df_cleaned.at[index, '斗泉识别截图'] = res["coinKindItemVersionInfo"]["itemName"] + "_" + \
                                                           res["coinKindItemVersionInfo"]['name']
                    print(f"{front_pic}:识别结果  {res["coinKindItemVersionInfo"]['name']}")
                    success += 1
                elif res['coinKindItemVersionInfo'] is None and res['linkUrl'] != "":
                    df_cleaned.at[index, '斗泉识别截图'] = res['linkUrl']
                else:
                    error += 1
        except Exception as e:
            error += 1
            print(f"{front_pic}   err:{e}")
            continue
    print(f"一共{total},成功{success},失败{error} \n")
df_cleaned.to_csv(csv_path, index=False)
