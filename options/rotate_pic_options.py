import argparse


def parse():
    parser = argparse.ArgumentParser(prog="旋转图片", description="execl钱币数据加载")
    parser.add_argument("--rotate_pic_path_list", required=True, type=str, help="需要旋转的列表")
    parser.add_argument("--rotate_angle", default=5, type=int, help="旋转角度")
    parser.add_argument("--max_num", default=1000, type=int, help="文件夹内最多保留多少张")
    parser.add_argument("--max_threads_num", default=10, type=int, help="处理图片最大线程数")
    return parser.parse_args()
