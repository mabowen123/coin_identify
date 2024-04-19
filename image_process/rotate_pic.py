# -*- coding: utf-8 -*-
import math
import os
import sys
import concurrent.futures
import random

from PIL import Image

sys.path.append("../")
from tool import file
from options import rotate_pic_options as rotatePicOpt
from tool.print import print_with_timestamp


class RotatePic():
    def __init__(self, params):
        self.rotate_pic_path_list = params.rotate_pic_path_list.split(",")
        self.max_num = params.max_num
        self.rotate_angle = int(params.rotate_angle)
        self.max_threads_num = params.max_threads_num
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads_num)
        self.del_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads_num)

    def rotate(self):
        for dir_path in self.rotate_pic_path_list:
            os_list = os.listdir(dir_path)
            for pic_path in os_list:
                if not pic_path.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    continue
                if "rotate" in pic_path.lower():
                    continue
                self.executor.submit(self.process, os.path.join(dir_path, pic_path))
        self.executor.shutdown(wait=True)

    def process(self, pic_path):
        print_with_timestamp(f"开始旋转{pic_path}")
        file_extension = os.path.splitext(pic_path)[1]
        rotate_num = math.ceil(360 / self.rotate_angle)
        for i in range(rotate_num):
            angle = (i + 1) * self.rotate_angle
            new_pic_path = pic_path.replace(file_extension, f"_rotate_angle_{angle}{file_extension}")
            img = Image.open(pic_path)
            rotated_img = img.rotate(angle)
            rotated_img.save(new_pic_path)

    def del_redundant_pic(self):
        for dir_path in self.rotate_pic_path_list:
            os_list = os.listdir(dir_path)
            os_list = [path for path in os_list if "_rotate_angle_" in path]
            dir_path_len = len(os_list)
            if dir_path_len < self.max_num:
                continue
            for i in range(3):
                random.shuffle(os_list)
            for os_item in os_list[self.max_num:]:
                self.del_executor.submit(file.remove_file, os.path.join(dir_path, os_item), False)
        self.del_executor.shutdown(wait=True)


if __name__ == "__main__":
    params = rotatePicOpt.parse()
    print("\n")
    print_with_timestamp(f"加载旋转图片输入参数: {params}")
    rotatePic = RotatePic(params)
    rotatePic.rotate()
    rotatePic.executor.shutdown(wait=True)
    rotatePic.del_redundant_pic()
    print_with_timestamp(f"处理结束")
