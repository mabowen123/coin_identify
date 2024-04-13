import os
import sys

sys.path.append("../")
from tool import file
import pandas as pd
from tool.print import print_with_timestamp


def map_execl_to_load_image(file_path):
    df = pd.read_excel(file_path, header=0, index_col=False)
    duplicate_subset = "正面图片"
    is_duplicate = df.duplicated(subset=[duplicate_subset])
    if is_duplicate.any():
        print_with_timestamp(file_path)
        print_with_timestamp("-------------------有重复行----------------------")
        count_per_image = df.groupby(duplicate_subset).size().reset_index(name='counts')
        images_with_more_than_two = count_per_image[count_per_image['counts'] > 1]
        for index, row in images_with_more_than_two.iterrows():
            print(f"图片: {row['正面图片']}, 出现次数: {row['counts']}")
        print_with_timestamp("-------------------有重复行----------------------")
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

    return res, df.shape[0]


def load_and_check_file(file_path):
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
    for column in required_columns:
        if column not in columns:
            print_with_timestamp(f"-------{file_name} 缺少以下必须字段: {column}")
            exit()

    print_with_timestamp(f"-------{file_name} 校验成功")


def merge_execl(train_execl_path, execl_merge_path):
    paths = os.listdir(execl_merge_path)
    original_df = pd.read_excel(train_execl_path)
    for path in paths:
        full_path = os.path.join(execl_merge_path, path)
        if not full_path.endswith('.xlsx'):
            continue
        target_df = pd.read_excel(full_path, header=0)
        original_df = original_df._append(target_df, ignore_index=True)

    with pd.ExcelWriter(train_execl_path, engine='xlsxwriter') as writer:
        original_df.to_excel(writer, index=False)
    df = pd.read_excel(train_execl_path)
    deduplicated_df = df.drop_duplicates()
    deduplicated_df.to_excel(train_execl_path, index=False)
