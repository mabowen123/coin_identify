import argparse


def parse():
    parser = argparse.ArgumentParser(prog="验证服务", description="验证服务")
    parser.add_argument("--model_path", required=True, type=str, help="模型路径")
    parser.add_argument("--index_path", required=True, type=str, help="索引路径")
    return parser.parse_args()
