import os.path

import pandas as pd

train = "/Users/mabowen/Documents/coin_sample/崇宁通宝/4.15崇宁通宝训练集.xlsx"
vaild = "/Users/mabowen/Documents/coin_sample/崇宁通宝/4.15崇宁通宝测试集.xlsx"
# 对比a、b两个文件的正面图片是否有重复
dfA = pd.read_excel(vaild)
dfB = pd.read_excel(train)
# 查找正面图片重复的行
repeats = dfA[dfA['正面图片'].isin(dfB['正面图片'])]
# 按大类别分组计数
repeats_count = repeats.groupby('大类别').size()
for big_class, count in repeats_count.to_dict().items():
    print(f"类别:{big_class} 重复数量:{count}")

# 打印重复行与被重复行的证书编号、正面图片、大类别
for index, row in repeats[['证书编号', '正面图片', '大类别']].iterrows():
    print(f"大类别:{row['大类别']} 正面图片: {row['正面图片']}")

print("\n")
paths = [train, vaild]
dict_data = {}
all_key = []
for path in paths:
    basename = os.path.basename(path)
    df = pd.read_excel(path)
    duplicate_subset = "正面图片"
    is_duplicate = df.duplicated(subset=[duplicate_subset], keep=False)
    if is_duplicate.any():
        duplicate_rows = df[is_duplicate]
        count_per_image = df.groupby(duplicate_subset).size().reset_index(name='counts')
        images_with_more_than_two = count_per_image[count_per_image['counts'] > 1]
        for index, row in images_with_more_than_two.iterrows():
            print(f"图片: {row['正面图片']}, 出现次数: {row['counts']}")
    col_data = df["大类别"]
    val_count = col_data.value_counts().to_dict()
    print(f"{basename} 类别个数:{len(val_count)}")
    for key in val_count.keys():
        if key not in all_key:
            all_key.append(key)
    dict_data[basename] = val_count


res = {}
print("\n")
for key in all_key:
    if key not in res:
        res[key] = {}
    for file_name, dict_item in dict_data.items():
        res[key][file_name] = dict_item.get(key, 0)

for key, item in res.items():
    if 0 in item.values():
        print({key: item})
