# -*- coding: utf-8 -*-
import concurrent
import math
import os
import random
import subprocess
import sys
import concurrent.futures
import time

sys.path.append("../")

from options import divide_options as divideOpt
from tool import file
from tool.print import print_with_timestamp
import timm


class DivideData():
    def __init__(self, params):
        self.divide_path = params.divide_path
        self.split_ratio = params.split_ratio
        self.max_threads_num = params.max_threads_num
        self.train_min_num = params.train_min_num
        self.train_num_too_short = []
        self.file_path = os.path.join(self.divide_path, "origin")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads_num)
        [self.train_datasets_path, self.test_datasets_path] = self.mk_output_dir()

    def mk_output_dir(self):
        paths = [
            os.path.join(self.divide_path, "train"),
            os.path.join(self.divide_path, "test"),
        ]
        file.mkdir_list(paths)
        return paths

    def save_index(self):
        b = timm.data.create_dataset(name="test", root=self.train_datasets_path)
        with open(os.path.join(self.divide_path, "index.txt"), 'w') as f:
            for k, v in b.reader.class_to_idx.items():
                f.write(str(k) + "\n")

    def divide_datasets(self):
        dirs = os.listdir(self.file_path)
        for label_name in dirs:
            dir_path = os.path.join(self.file_path, label_name)
            if os.path.isdir(dir_path):
                self.executor.submit(self.process, dir_path, label_name)

    def print_train_num_too_short_list(self):
        if len(self.train_num_too_short) > 0:
            print_with_timestamp("------------------样本过少的文件路径------------------")
            str_result = ','.join(str(x) for x in self.train_num_too_short)
            print_with_timestamp(str_result)
            print_with_timestamp("-------------------------------------------------")

            # 要执行的 Python 脚本的文件路径
            script_directory = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_directory, "./rotate_pic.py")
            # 使用 subprocess 模块运行另一个 Python 脚本
            try:
                subprocess.run(["python3", script_path, "--rotate_pic_path_list", str_result], check=True)
                self.divide_datasets()
            except subprocess.CalledProcessError as e:
                print(f"脚本执行失败: {e}")

    def process(self, dir_path, label_name):
        os_list = os.listdir(dir_path)
        if len(os_list) == 0:
            print_with_timestamp(f"------------------{dir_path}是空文件夹------------------")
            return

        os_list = file.image_verify(os_list)
        for i in range(3):
            random.shuffle(os_list)
        len_origin_os = len(os_list)
        len_train_os = int(len_origin_os * self.split_ratio)
        print_with_timestamp(
            f"开始切割 {label_name},"
            f"共有张{len_origin_os}图片,"
            f"移动训练(train)集到{len_train_os}张,"
            f"移动测试(val)集到{len_origin_os - len_train_os}张"
        )
        train_list = os_list[:len_train_os]
        if len_train_os < self.train_min_num:
            self.train_num_too_short.append(dir_path)

        file.file_list_copy(train_list, dir_path, os.path.join(self.train_datasets_path, label_name))
        test_list = os_list[len_train_os:]
        file.file_list_copy(test_list, dir_path, os.path.join(self.test_datasets_path, label_name))


if __name__ == "__main__":
    params = divideOpt.parse()
    print("\n")
    print_with_timestamp(f"切割数据 输入参数: {params}")
    divide = DivideData(params)
    divide.divide_datasets()
    divide.print_train_num_too_short_list()
    os.chmod(divide.divide_path, 0o777)
    divide.executor.shutdown(wait=True)
    time.sleep(3)
    divide.save_index()
