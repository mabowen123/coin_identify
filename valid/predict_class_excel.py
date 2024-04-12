import argparse
import json
import logging
import os
import subprocess
import time
import numpy as np

import pandas as pd
import requests

log_path = os.path.join(os.path.dirname(__file__), "search.log")
handler = logging.FileHandler(log_path, encoding="UTF-8")
console = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(lineno)d | %(message)s"
)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger = logging.getLogger("download")
logger.setLevel(level=logging.DEBUG)
logger.addHandler(handler)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=False, help="data csv path")
parser.add_argument(
    "--suffix", default=".jpg", help="save image suffix, option: [.png, .jpg, .jpeg]"
)
parser.add_argument("--thread", type=int, default=4, help="thread nums, defualt 4")
parser.add_argument("--process", type=int, default=4, help="process nums, default 4")
parser.add_argument("--url", default="", help="image download url")
parser.add_argument("--dst", default="yanshouji_data", help="dst save path")
parser.add_argument("--per", type=int, default=1000, help="per process deal image nums")
parser.add_argument(
    "--col", type=str, default="图片", help="csv column name (download url)"
)
parser.add_argument("--sample", type=int, help="sample")

# todo: 最新添加
parser.add_argument("--excel_path", type=str, default="", help="excel path")
parser.add_argument("--check_big_label_name", type=str, default="", help="check big label name")
parser.add_argument("--req_type", type=str, default="local", help="http req type")
parser.add_argument("--url_route", type=str, default="aaa", help="url route")
parser.add_argument("--serial_number_is_null", type=str, default="false", help="serial number is null")
args = parser.parse_args()


# import sys

# root_path = "/gpu2-data/huxiuyun/pytorch-image-models//old_coin_code"
# sys.path.append(root_path)

# 创建ONNX Runtime推理会话
# model_path = '/gpu2-data/huxiuyun/pytorch-image-models/output/train/20230814-142628-efficientnet_b5-416/model_best.onnx'
# ort_session = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# model_path2 = '/gpu2-data/huxiuyun/pytorch-image-models/output/train/20230810-152903-densenet121-224/model_best.onnx'
# ort_session2 = onnxruntime.InferenceSession(model_path2,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


# # todo: 特殊逻辑 一般用不上
# id_name_dict = {}
# excel_path_id = "/gpu4-data/datasets/paper_coin_data/America/美国纸币币种id映射表.xlsx"
# xls = pd.read_excel(excel_path_id, sheet_name=None)
# id_name_dict = {}
# for sheet_name, df in xls.items():
#     for row in df.index:
#         item = df.loc[row]
#         id_1 = str(item["正面id"])
#         id_2 = str(item["背面id"])
#         label_name = item["纸币币种编号"]
#         id_name_dict[id_1] = label_name
#         id_name_dict[id_2] = label_name

# print(len(id_name_dict))


def thread_download3_local(url1):
    predict_label_c = ["", "", "", "", ""]
    predict_score1 = [0, 0, 0, 0, 0]
    try:
        start_time = time.time()
        url_class1 = "http://127.0.0.1:86/classifyc/api/predict_old?pic1=" + url1
        # print(url_class1)
        r = requests.get(url_class1)
        res_js = json.loads(r.content)
        if "data" not in res_js:
            label1 = ""
            # predict_label_c = ["", "", "", "", ""]
            # predict_score1 = [0, 0, 0, 0, 0]
        else:
            predict_label_c = res_js["data"]
            predict_score1 = res_js["cls_score"]
            label1 = predict_label_c[0]

        # print('loacl:', predict_label_c, predict_score1)
        return predict_label_c, predict_score1

    except (Exception,) as e:
        print(e)
        logger.error(
            f"processing {url1} error, pi: {1}, ti{1}, error: {e}", exc_info=True
        )
        raise ValueError(str("url download err"))
        return predict_label_c, predict_score1


def thread_download3_online(url1):
    predict_label_c = ["", "", "", "", ""]
    predict_score1 = [0, 0, 0, 0, 0]
    try:
        start_time = time.time()
        # todo: 可根据实际情况改动
        url_class1 = (
                fr"http://imgt.wpt.la/ancient-coin/api/{url_route}?pic=" + url1
        )
        # print(url_class1)

        r = requests.get(url_class1)
        res_js = json.loads(r.content)
        if "data" not in res_js:
            print(url_class1)
            label1 = ""
            # predict_label_c = ["", "", "", "", ""]
            # predict_score1 = [0, 0, 0, 0, 0]
        else:
            predict_label_c1 = res_js["others"]["coin_list"]
            predict_score1 = res_js["others"]["score_list"]
            label1 = predict_label_c1

        # print('online:', predict_label_c, predict_score1)
        return predict_label_c1, predict_score1

    except (Exception,) as e:
        print(e)
        logger.error(
            f"processing {url1} error, pi: {1}, ti{1}, error: {e}", exc_info=True
        )
        raise ValueError(str("url download err"))
        return predict_label_c, predict_score1


def main():
    xls = pd.read_excel(excel_path, sheet_name=None)
    no_process = 0
    right, error = 0, 0

    # 统计类别的数据占比
    label_dist = {}
    for sheet_name, df in xls.items():
        # print("Sheet Name:", sheet_name)
        # print("Index:", df.index)

        for row in df.index:
            item = df.loc[row]

            label_name = str(item["大类别"])
            big_label = str(item["钱币名称"])
            # 组装为 '钱币名称 + 面值 + 大类别'的格式
            face_val = str(item["面值"])
            # 举例：finish_name=景德元宝_小平_隔轮
            finish_name = fr'{big_label}_{face_val}_{label_name}'
            # print("---> finish_name:", finish_name)

            # 过滤掉没有证书的数据数据---> 特殊业务逻辑
            if serial_number_is_null == 'true' and not pd.isnull(item["证书编号"]):
                # print('跳过---> item["证书编号"]=%s' % (item["证书编号"]))
                continue

            # 比如check_big_label_name = '景德元宝'
            # 如果finish_name包含待过滤的字符 那么就不跳过
            if check_big_label_name is not None and not check_big_label_name in finish_name:
                # print('跳过---> 命中check_big_label_name in finish_name逻辑! %s %s' % (check_big_label_name, finish_name))
                continue

            if finish_name not in label_dist or label_dist[finish_name] is None:
                label_dist[finish_name] = {'right': 0, 'error': 0, 'no_process': 0, 'all': 0}

            label_dist[finish_name]["all"] += 1
            url1 = item["正面图片"]
            # print("---> label_dist 0:", label_dist[finish_name])
            # print("---> url1:", url1)
            if req_type == 'local':
                predict_label_c1, predict_score1 = thread_download3_local(url1)
            else:
                predict_label_c1, predict_score1 = thread_download3_online(url1)
                # print(predict_label_c1, predict_score1)
            predict_name = predict_label_c1[0]
            # print("predict_name:", predict_name)
            for it in predict_label_c1:
                if check_big_label_name in it:
                    predict_name = it
                    break

            # print("---> label_dist 1:", label_dist[finish_name])
            # print("predict_name:", predict_name)
            if predict_name == "":
                label_dist[finish_name]["no_process"] += 1
                no_process += 1
                continue
            if label_name in predict_name:
                label_dist[finish_name]["right"] += 1
                right += 1
            else:
                label_dist[finish_name]["error"] += 1
                error += 1
                print(
                    label_name in predict_name,
                    label_name,
                    predict_label_c1[:2],
                    predict_score1[:2],
                    url1,
                )
            # print("---> label_dist 2:", label_dist[finish_name])

            # print(f'result: right={right},  all={right+error}, percentage={right/(right+error+1)}')
    # print(
    #     f"result: right={right},  all={right+error}, percentage={right/(right+error)}"
    # )
    for finish_name, values in label_dist.items():
        print(
            f"类别={finish_name}, 预测正确={values['right']}, percentage={values['right'] / (values['all'])}, no_process={values['no_process']}, 样本量={values['all']}"
        )

    print(
        f"result: right={right},  all={right + error}, percentage={right / (right + error + 1)}"
    )
    print("no_process: ", no_process)

    print("----- post all ---------")


if __name__ == "__main__":
    global excel_path
    global check_big_label_name
    global url_route

    check_big_label_name = args.check_big_label_name
    req_type = args.req_type
    excel_path = args.excel_path
    url_route = args.url_route
    serial_number_is_null = args.serial_number_is_null
    index_path = args.index_path
    model_path1 = args.model_path1

    # 要执行的 Python 脚本的文件路径
    script_directory = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_directory, "./rotate_pic.py")
    # 使用 subprocess 模块运行另一个 Python 脚本
    try:
        subprocess.run(["python3", script_path, "--index_path", index_path, "--model_path", model_path1], check=True,
                       capture_output=False)
        main()
    except subprocess.CalledProcessError as e:
        print(f"脚本执行失败: {e}")
