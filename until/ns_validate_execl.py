import os

import pandas as pd

index_execl_path = "/Users/mabowen/Downloads/南宋-样本/嘉熙通宝映射表.xlsx"
validate_execl_path = "/Users/mabowen/Downloads/南宋-样本/嘉熙通宝训练集.xlsx"
train_execl_path = "/Users/mabowen/Downloads/南宋-样本/嘉熙通宝测试集.xlsx"

# 读取映射表 正反面特征
index_execl_df = pd.read_excel(index_execl_path)
front_feature = index_execl_df["正面特征"].to_list()
back_feature = index_execl_df["反面特征"].to_list()
map_version = set(index_execl_df["版别ID"].to_list())

# 读取验证集的版别分类是不是和训练集相等
paths = [validate_execl_path, train_execl_path]
dict_data = {}
all_key = []
front_pic_dict = {}
back_pic_dict = {}

version_data_set = set([])
for inx, path in enumerate(paths):
    basename = os.path.basename(path)
    df = pd.read_excel(path)
    for index, item in df.iterrows():
        feature = back_feature
        if item['版别位置'] == "正面":
            feature = front_feature
            if item["正面图片"] not in front_pic_dict:
                front_pic_dict[item["正面图片"]] = 1
            else:
                front_pic_dict[item["正面图片"]] += 1
        elif item["版别位置"] == "背面":
            if item["反面图片"] not in back_pic_dict:
                back_pic_dict[item["反面图片"]] = 1
            else:
                back_pic_dict[item["反面图片"]] += 1
        else:
            print(f"---- {basename} 出现 版别位置除了正面和背面的数据 {item['版别位置']} ----")

        if "weipaitang" not in item["正面图片"] or "weipaitang" not in item['反面图片']:
            print("-------存在不属于wpt的域名图片---------")

        if item["版别分类"] not in feature:
            print(f"---- {basename} 出现不存在映射表的版别分类 {item['版别分类']} ----")

    col_data = df["版别分类"]
    if inx == 1:
        tmp_version = df["版别ID"]
        version_data_set = version_data_set | set(tmp_version.to_list())

    val_count = col_data.value_counts().to_dict()
    print(f"{basename} 类别个数:{len(val_count)}")
    for key in val_count.keys():
        if key not in all_key:
            all_key.append(key)
    dict_data[basename] = val_count

# mvDiff = map_version - version_data_set
# if mvDiff:
#     print("以下版别在映射中存在 测试版别不存在", mvDiff)

vmDiff = version_data_set - map_version
if vmDiff:
    print("以下版别在测试版别存在 映射版别不存在", vmDiff)

for pic in set(back_pic_dict.keys()).intersection(set(front_pic_dict.keys())):
    print(f"---- {pic} 既存在正面图内也存在反面图内 ----")

for pic, count in back_pic_dict.items():
    if count > 1:
        print(f"---- {pic} 反面图出现一共 {count} 次----")

print("\n")
for pic, count in front_pic_dict.items():
    if count > 1:
        print(f"---- {pic} 正面图出现一共 {count} 次 ----")

res = {}
print("\n")

# print("版别分类 all_key=", all_key)

# print("dict_data=",dict_data)
for key in all_key:
    if key not in res:
        res[key] = {}
    for file_name, dict_item in dict_data.items():
        res[key][file_name] = dict_item.get(key, 0)

for key, item in res.items():
    if 0 in item.values():
        print({key: item})
