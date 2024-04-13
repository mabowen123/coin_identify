# -*- coding: utf-8 -*-
import concurrent
import os
import shutil
import sys
import concurrent.futures
import cv2

sys.path.append("../")

import load_execl
import image_until
from options import image_options as imgOpt
from tool import file
from tool.print import print_with_timestamp


class LoadImageData():
    def __init__(self, params):
        # 输出目录路径
        self.output_path = params.output_path
        # 训练集目录路径
        self.train_execl_path = params.train_execl_path
        # 测试集目录路径
        self.valida_execl_path = params.valida_execl_path
        # 最大线程数
        self.max_threads_num = params.max_threads_num
        # 判断是否可合并execl
        self.merge_execl()
        # 前置校验execl文件
        self.load_and_check_file()
        # 创建出 原图/验证的文件夹
        [self.data_sets_origin_path, self.data_sets_valida_path] = self.mk_output_dir()
        # 训练集多少张
        self.train_data_total = 0
        self.train_data_process_total = 0
        # 验证集多少张
        self.valida_data_total = 0
        self.valida_data_process_total = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads_num)

    def merge_execl(self):
        execl_merge_path = os.path.join(self.output_path, "execl_merge")
        if not file.file_exists(execl_merge_path):
            return

        if len(os.listdir(execl_merge_path)) > 0:
            file_extension = os.path.splitext(self.train_execl_path)[1]
            new_path = self.train_execl_path.replace(file_extension, f"_backup{file_extension}")
            if not file.file_exists(new_path):
                shutil.copyfile(self.train_execl_path, new_path)
            load_execl.merge_execl(self.train_execl_path, execl_merge_path)
            print_with_timestamp("execl合并完成")

    # 获取execl 数据 分发线程处理
    def process_data(self):
        # 处理训练集execl 获取数据 创建文件夹
        train_data, self.train_data_total = load_execl.map_execl_to_load_image(self.train_execl_path)
        valida_data, self.valida_data_total = load_execl.map_execl_to_load_image(self.valida_execl_path)
        res = {
            self.data_sets_origin_path: train_data,
            self.data_sets_valida_path: valida_data,
        }
        print_with_timestamp("------------------开始处理图片------------------")
        for path, label_item in res.items():
            for label_name, coin_data in label_item.items():
                # 训练 or 测试 路径 创建对应的文件夹
                label_name_path = os.path.join(path, label_name)
                file.mkdir(label_name_path)
                # 线程处理
                for coin_item in coin_data:
                    self.executor.submit(self.threads_process, coin_item, label_name_path)
        # 释放线程
        self.executor.shutdown(wait=True)
        self.print_process_log(True)
        print_with_timestamp("------------------处理图片结束------------------")

    def print_process_log(self, need_print):
        if need_print:
            process_data_total = self.get_process_data_total()
            print_with_timestamp(
                f"累计处理图片{process_data_total},一共{self.get_data_total()}个; "
                f"处理训练图片{self.train_data_process_total}个,一共{self.train_data_total}; "
                f"处理验证图片{self.valida_data_process_total}个,一共{self.valida_data_total}; "
            )

    #  增加处理个数
    def incr_process_total(self, label_name_path):
        if "valida" in label_name_path:
            self.valida_data_process_total += 1
        elif "origin" in label_name_path:
            self.train_data_process_total += 1
        self.print_process_log((self.get_process_data_total() % 100 == 0))

    # 获取已经处理的数据条数
    def get_process_data_total(self):
        return self.valida_data_process_total + self.train_data_process_total

    # 获取总共需要处理的条数
    def get_data_total(self):
        return self.train_data_total + self.valida_data_total

    # 线程处理
    def threads_process(self, coin_item, label_name_path):
        url = coin_item["正面图片"]
        try:
            save_path = image_until.save_image(url, coin_item["版别"], label_name_path)
            if file.file_exists(save_path, False):
                image_until.image_gray(save_path)
                scale_img, is_ok = image_until.scale_img(save_path)
                if is_ok:
                    cv2.imwrite(save_path, scale_img)
                    self.incr_process_total(label_name_path)
                else:
                    file.remove_file(save_path, True)
        except Exception as e:
            print_with_timestamp("异常图片url: %s err: %s" % (url, e))

    # 前置校验
    def load_and_check_file(self):
        paths = [
            self.train_execl_path,
            self.valida_execl_path
        ]
        for path in paths:
            load_execl.load_and_check_file(path)

    # 新建输出文件
    def mk_output_dir(self):
        data_sets_origin_path = os.path.join(self.output_path, "origin")
        data_sets_valida_path = os.path.join(self.output_path, "valida")
        paths = [data_sets_origin_path, data_sets_valida_path]
        file.mkdir_list(paths)
        return [data_sets_origin_path, data_sets_valida_path]


if __name__ == "__main__":
    params = imgOpt.parse()
    print("\n")
    print_with_timestamp(f"加载图片输入参数: {params}")
    load_image = LoadImageData(params)
    load_image.process_data()
    os.chmod(load_image.output_path, 0o777)
