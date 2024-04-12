import argparse


def parse():
    parser = argparse.ArgumentParser(prog="图片加载", description="execl钱币数据加载")
    parser.add_argument("--train_execl_path", required=True, type=str, help="需要加载的训练数据execl路径")
    parser.add_argument("--valida_execl_path", required=True, type=str, help="需要加载的验证数据execl路径")
    parser.add_argument("--output_path", required=True, type=str, help="图片输出路径")
    parser.add_argument("--max_threads_num", default=50, type=int, help="处理图片最大线程数")
    return parser.parse_args()
