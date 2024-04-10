import os
import sys
import time
import random
import requests

sys.path.append("../")
import load_execl
from options import image_options as imgOpt
from tool import file


class LoadImageData():
    def __init__(self, params):
        # 输出目录路径
        self.output_path = params.output_path
        # 训练集目录路径
        self.train_execl_path = params.train_execl_path
        # 测试集目录路径
        self.test_execl_path = params.test_execl_path
        # 前置校验execl文件
        self.load_and_check_file()
        # 创建出 训练/测试/验证的文件夹
        [self.data_sets_train_path, self.data_sets_valid_path] = self.mk_output_dir()

    def process_data(self):
        # 处理训练集execl 获取数据 创建文件夹
        res = {
            self.data_sets_train_path: load_execl.map_execl_to_load_image(self.train_execl_path),
            self.data_sets_valid_path: load_execl.map_execl_to_load_image(self.test_execl_path),
        }

        for path, label_item in res.items():
            for label_name, coin_data in label_item.items():
                # 训练 or 测试 路径 创建对应的文件夹
                label_name_path = os.path.join(path, label_name)
                file.mkdir(label_name_path)
                # 线程处理
                for coin_item in coin_data:
                    start_time = time.time()
                    ti = random.randint(1, 10000000)
                    img_id = str(int(start_time * 10) + ti)
                    url = coin_item["正面图片"]
                    r = requests.get(url)
                    if r.status_code == 200:
                        img_path = os.path.join(str(label_name_path), "{}_{}.jpg".format(coin_item["版别"], img_id))
                        with open(
                                img_path,
                                "wb",
                        ) as f:
                            f.write(r.content)

    def load_and_check_file(self):
        paths = [
            self.train_execl_path,
            self.test_execl_path
        ]
        for path in paths:
            load_execl.load_and_check_file(path)

    def mk_output_dir(self):
        # 删除输出目录
        if not file.remove_file(self.output_path):
            exit()
        data_sets_train_path = os.path.join(self.output_path, "train")
        data_sets_valid_path = os.path.join(self.output_path, "valida")
        paths = [data_sets_train_path, os.path.join(self.output_path, "test"), data_sets_valid_path]
        for path in paths:
            file.mkdir(path)
            if not file.file_exists(path):
                exit()
        return [data_sets_train_path, data_sets_valid_path]

if __name__ == "__main__":
    params = imgOpt.parse()
    load_image = LoadImageData(params)
    load_image.process_data()
