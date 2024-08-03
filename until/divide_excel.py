import pandas as pd
import random

# 读取原始Excel文件
file_path = '/Users/mabowen/Downloads/嘉庆通宝/嘉庆通宝测试集730.xlsx'
df = pd.read_excel(file_path)

df = df[df["版别位置"] == "正面"]

# # 假设我们按照'Category'列进行分类
category_column = '版别分类'


categories = df[category_column].unique().tolist()


dd = categories.copy()
# 创建两个新的DataFrame用于存储划分后的数据
df1, df2 = pd.DataFrame(), pd.DataFrame()

# 按照7:3的比例划分数据
for category in categories:
    # 获取该类别的所有行
    subset = df[df[category_column] == category]

    # 计算需要划分的行数（70%和30%）
    total_rows = len(subset)
    split_point = int(total_rows * 0.7)  # 80%的分割点

    if split_point <= 1:
        split_point = total_rows
        df2 = pd.concat([df2, subset.iloc[:split_point]])
    else:
        df2 = pd.concat([df2, subset.iloc[split_point:]])

    # 随机划分以避免总是取前70%或后30%，如果需要顺序划分，可直接切片
    # random.shuffle(subset.index)  # 打乱索引以随机划分

    df1 = pd.concat([df1, subset.iloc[:split_point]])

# 保存到两个新的Excel文件
df1.to_excel('./嘉庆通宝_70%.xlsx', index=False)
df2.to_excel('./嘉庆通宝_30%.xlsx', index=False)

print("数据已按照7:3的比例划分并保存成功！")
