import os
import shutil
from pathlib import Path
from tool.print import print_with_timestamp


def remove_file(file_path, need_print=False):
    if not file_exists(file_path, False):
        need_print and print_with_timestamp(f"{file_path} 不存在不用删除")
        return True

    try:
        os.remove(file_path)
        need_print and print_with_timestamp(f"{file_path} 已经被删除.")
        return True
    except OSError as e:
        print_with_timestamp(f"删除错误: {file_path} : {e.strerror}")
        return False


def remove_dir(file_path):
    if not file_exists(file_path, False):
        print_with_timestamp(f"{file_path} 不存在不用删除")
        return True

    try:
        shutil.rmtree(file_path)
        print_with_timestamp(f"{file_path} 已经被删除.")
        return True
    except OSError as e:
        print_with_timestamp(f"删除错误: {file_path} : {e.strerror}")
        return False


def file_exists(file_path, need_print=True):
    try:
        exists = os.path.exists(file_path)
        need_print and print_with_timestamp(f"{file_path} 检查路径是否存在:{exists}")
        return exists
    except OSError as e:
        print_with_timestamp(f"查询路径存在错误: {file_path} : {e.strerror}")
        return False


def mkdir(file_path):
    Path(file_path).mkdir(parents=True, exist_ok=True)
    os.chmod(file_path, 0o777)


def mkdir_list(paths):
    for path in paths:
        if not remove_dir(path):
            exit()
        mkdir(path)
        if not file_exists(path, False):
            print_with_timestamp(f"{path} 创建失败")
            exit()
        print_with_timestamp(f"{path} 创建成功")


def image_verify(os_list):
    for item in os_list:
        if item.split(".")[-1] not in ["jpg", "png", "jpeg", "bmp", "JPG"]:
            os_list.remove(item)
    return os_list


def file_list_copy(file_list, src, dst):
    mkdir(dst)
    for file in file_list:
        src_path = os.path.join(src, file)
        dst_path = os.path.join(dst, file)
        shutil.copyfile(str(src_path), str(dst_path))