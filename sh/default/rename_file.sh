#!/bin/bash

# 检查命令行参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory=$1

# 检查目录是否存在
if [ ! -d "$directory" ]; then
    echo "Directory '$directory' does not exist."
    exit 1
fi

# 遍历目录中的文件
for file in "$directory"/*; do
    if [ -f "$file" ]; then  # 检查是否是文件
        filename=$(basename "$file")
        if [[ "$file" == *训练*  && "$filename" != 训练样本.${file##*.} &&  "$filename" != 训练样本_backup.${file##*.} ]]; then
            new_name="${file%/*}/训练样本.${file##*.}" # 替换文件名为 "训练样本"
            mv "$file" "$new_name"  # 重命名文件
            echo "Renamed '$file' to '${file%/*}/训练样本.${file##*.}'"
        fi

        if [[ "$file" == *测试* ]]; then  # 检查文件名中是否包含 "测试"
            new_name="${file%/*}/验证样本.${file##*.}" # 替换文件名为 "训练样本"
            mv "$file" "$new_name"  # 重命名文件
            echo "Renamed '$file' to '${file%/*}/验证样本.${file##*.}'"
        fi
    fi
done
echo "Renaming complete."
