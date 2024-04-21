index_path = "index.txt"
with open(index_path, 'r') as file:
    # 读取文件的所有行
    lines = file.readlines()

# 转化成字典
all_class_names_27 = {idx: string.strip() for idx, string in enumerate(lines)}
print(all_class_names_27)
