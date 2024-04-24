import os
import sys

sys.path.append("../")
from tool import file
import pandas as pd
from tool.print import print_with_timestamp


def map_execl_to_load_image(file_path, coin_type):
    df = pd.read_excel(file_path, header=0, index_col=False)
    res = dict()
    for index, row in df.iterrows():
        # 直接是execl 的 字典
        item = row.to_dict()
        if coin_type == "bs":
            big_class = item["大类别"]
            value = item["面值"]
            coin_class = str(item["钱币名称"]).replace("/", " ")
            label_name = coin_class + "_" + value + "_" + big_class
            item["label_name"] = label_name
        elif coin_type == 'ns':
            label_name = item["版别分类"]
            item["label_name"] = item["版别分类"]
        else:
            print_with_timestamp("异常的版别分类")
            exit()

        if label_name not in res:
            res[label_name] = []
        res[label_name].append(item)

    return res, df.shape[0]


def load_and_check_file(file_path, coin_type):
    # 判断文件路径是否存在
    if not file.file_exists(file_path):
        exit()

    # 读取execl文件
    xls = pd.ExcelFile(file_path)
    xls.close()

    file_name = os.path.basename(file_path)
    print_with_timestamp(f"-------{file_name} 读取文件")

    sheet_nums = len(xls.sheet_names)
    if sheet_nums > 1:
        print_with_timestamp(f"-------{file_name} 存在{sheet_nums}个sheet,确保只有一个sheet")
        exit()

    xls = pd.read_excel(file_path)
    columns = xls.columns.tolist()

    # 指定需要判断的字段列表
    required_columns = ["钱币名称", "正面图片", "面值", "大类别"]
    if coin_type == "ns":
        required_columns = ["钱币名称", "正面图片", "反面图片", "面值", "版别分类", "版别位置", "版别"]
    for column in required_columns:
        if column not in columns:
            print_with_timestamp(f"-------{file_name} 缺少以下必须字段: {column}")
            exit()

    print_with_timestamp(f"-------{file_name} 校验成功")


def merge_execl(train_execl_path, execl_merge_path):
    paths = os.listdir(execl_merge_path)
    original_df = pd.read_excel(train_execl_path)
    concat = [
        original_df
    ]
    original_df_col = original_df.columns.tolist()
    for path in paths:
        full_path = os.path.join(execl_merge_path, path)
        if not full_path.endswith('.xlsx'):
            continue
        print_with_timestamp(f"合并{full_path}到训练文件:{train_execl_path}")
        target_df = pd.read_excel(full_path, header=0)
        if original_df_col != target_df.columns.tolist():
            target_df = target_df.reindex(columns=original_df_col)
        concat.append(target_df)
    pd.concat(concat, ignore_index=True).drop_duplicates().to_excel(train_execl_path, index=False)


def get_coin_type(output_path):
    # 北宋
    default_type = "bs"
    if file.file_exists(os.path.join(output_path, "index_map.xlsx")):
        # 南宋
        default_type = "ns"

    return default_type
