import os
import sys

sys.path.append("../")

from tool import file
import pandas as pd


def map_execl_to_load_image(file_path):
    df = pd.read_excel(file_path, header=0, index_col=False)
    res = dict()
    for index, row in df.iterrows():
        # 直接是execl 的 字典
        item = row.to_dict()
        big_class = item["大类别"]
        value = item["面值"]
        coin_class = str(item["钱币名称"]).replace("/", " ")
        label_name = coin_class + "_" + value + "_" + big_class
        item["label_name"] = label_name
        if label_name not in res:
            res[label_name] = []
        res[label_name].append(item)
    return res


def load_and_check_file(file_path):
    # 判断文件路径是否存在
    if not file.file_exists(file_path):
        exit()

    # 读取execl文件
    xls = pd.ExcelFile(file_path)
    xls.close()

    file_name = os.path.basename(file_path)
    print(f"-------{file_name} 读取文件")

    sheet_nums = len(xls.sheet_names)
    if sheet_nums > 1:
        print(f"-------{file_name} 存在{sheet_nums}个sheet,确保只有一个sheet")
        exit()

    xls = pd.read_excel(file_path)
    columns = xls.columns.tolist()

    # 指定需要判断的字段列表
    required_columns = ["钱币名称", "正面图片", "面值", "大类别"]
    for column in required_columns:
        if column not in columns:
            print(f"-------{file_name} 缺少以下必须字段: {column}")
            exit()

    print(f"-------{file_name} 校验成功")
