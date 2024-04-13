import argparse


def parse():
    parser = argparse.ArgumentParser(prog="验证execl", description="验证execl")
    parser.add_argument("--valida_execl_path", required=True, type=str, help="验证execl路径")
    parser.add_argument("--url", default="", type=str, help="请求链接如果为空就是本地连接")
    parser.add_argument("--model_path", default="", type=str, help="验证服务的模型")
    parser.add_argument("--index_path", default="", type=str, help="默认是验证execl路径下的index.txt")
    return parser.parse_args()
