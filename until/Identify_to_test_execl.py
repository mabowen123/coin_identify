import pandas as pd

path = "/Users/mabowen/Downloads/古钱币识别记录.csv"
index_map_path = "/Users/mabowen/Downloads/顺治通宝/顺治通宝映射表527.xlsx"
if index_map_path != "":
    df = pd.read_csv(path)
    index_df = pd.read_excel(index_map_path)
    index_df_dict = {}
    for index, item in index_df.iterrows():
        index_df_dict[item["版别ID"]] = item.to_dict()

    new_data = []
    for index, item in df.iterrows():
        index_df = index_df_dict[item["版别ID"]]
        front_data = [
            item["币种"],
            "",
            item["实物正面图片"],
            item["实物背面图片"],
            "",
            item["修改后面值"],
            index_df["版别"],
            item["修改后铸造局"],
            "正面",
            index_df["正面特征"],
            index_df["正面特征"],
            index_df["反面特征"],
            item["版别ID"]
        ]
        back_data = front_data.copy()
        back_data[8] = "背面"
        back_data[9] = index_df["反面特征"]
        new_data.append(front_data)
        new_data.append(back_data)
    last_df = pd.DataFrame(new_data,
                           columns=["钱币名称", "证书编号", "正面图片", "反面图片", "书体", "面值", "版别", "铸造局",
                                    "版别位置",
                                    "版别分类", "正面特征", "背面特征", "版别ID"])
    last_df.to_excel("test.xlsx", index=False)
else:
    df = pd.read_csv(path)
    new_data = []
    for index, item in df.iterrows():
        is_error = False
        if item["审核结果"] == "错误":
            is_error = True

        if is_error == True and (pd.isna(item["修改后版别"]) or pd.isna(item["修改后书体"])):
            continue

        if is_error == False and (pd.isna(item["版别"]) or pd.isna(item["书体"])):
            continue

        new_data.append([
            item["币种"],
            str(item["修改后版别"]) + str(item["修改后书体"]) if is_error else str(item["版别"]) + str(item["书体"]),
            "",
            item["实物正面图片"],
            item["修改后面值"] if is_error else item["面值"],
            str(item["修改后版别"]) + "_" + str(item["修改后书体"]) if is_error else str(item["版别"]) + "_" + str(
                item["书体"]),
        ])
    last_df = pd.DataFrame(new_data, columns=["钱币名称", "版别", "证书编号", "正面图片", "面值", "大类别"])
    last_df.to_excel("test.xlsx", index=False)
