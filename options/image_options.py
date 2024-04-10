import argparse


def parse():
    parser = argparse.ArgumentParser(prog="图片加载", description="execl钱币数据加载")
    parser.add_argument("--train_execl_path", required=True, type=str, help="需要加载的训练数据execl路径")
    parser.add_argument("--test_execl_path", required=True, type=str, help="需要加载的测试数据execl路径")
    parser.add_argument("--output_path", required=True, type=str, help="图片输出路径")
    return parser.parse_args()
