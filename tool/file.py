import os
import shutil
from pathlib import Path


def remove_file(file_path):
    if not file_exists(file_path):
        return True

    try:
        shutil.rmtree(file_path)
        print(f"{file_path} 已经被删除.")
        return True
    except OSError as e:
        print(f"删除错误: {file_path} : {e.strerror}")
        return False


def file_exists(file_path):
    try:
        exists = os.path.exists(file_path)
        print(f"{file_path} 检查路径是否存在:{exists}")
        return exists
    except OSError as e:
        print(f"查询路径存在错误: {file_path} : {e.strerror}")
        return False


def mkdir(file_path):
    return Path(file_path).mkdir(parents=True, exist_ok=True)

