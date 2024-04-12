import argparse


def parse():
    parser = argparse.ArgumentParser(prog="切割文件", description="切割数据到验证集和测试集")
    parser.add_argument("--divide_path", required=True, type=str, help="需要切割文件的路径")
    parser.add_argument("--split_ratio", default=0.8, type=str, help="留存多少在训练集省下都是测试集")
    parser.add_argument("--max_threads_num", default=20, type=int, help="处理图片最大线程数")
    parser.add_argument("--train_min_num", default=150, type=int, help="训练最小数量")
    return parser.parse_args()
