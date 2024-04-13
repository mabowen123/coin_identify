import json
import os
import subprocess
import sys
import time

import requests

sys.path.append("../")
from tool import netstat
from image_process import load_execl
from options import predict_execl_options as peOpt
from tool.print import print_with_timestamp


class PredictExecl():
    def __init__(self, param):
        self.valida_execl_path = param.valida_execl_path
        self.model_path = param.model_path
        self.index_path = param.index_path
        self.index_path = param.index_path if param.index_path else os.path.join(
            os.path.dirname(self.valida_execl_path), "index.txt")
        self.url = param.url
        # self.need_run_valida_server()

    def need_run_valida_server(self):
        if not self.is_local():
            return True

        netstat.kill_process_using_port(86)
        try:
            script_directory = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_directory, "../classifyc/run.py")
            subprocess.Popen(
                ["nohup", "python3", script_path, "--index_path", self.index_path, "--model_path", self.model_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True)
        except subprocess.CalledProcessError as e:
            print_with_timestamp(f"验证服务启动异常: {e}")
            exit()

        for i in range(10):
            time.sleep(2)
            try:
                r = requests.get('http://127.0.0.1:86/health')
                if r.status_code == 200:
                    return True
            except (Exception,):
                pass
        print_with_timestamp("验证服务启动异常,从sh/nohup.log 查看日志")
        exit()

    def is_local(self):
        return self.url == ""

    def get_req_url(self, image_url):
        url = f"http://127.0.0.1:86/classifyc/api/predict_old?pic1={image_url}"
        if not self.is_local():
            url = f"http://imgt.wpt.la/ancient-coin/api/{self.url}?pic={image_url}"
        return url

    def valida(self):
        res, total = load_execl.map_execl_to_load_image(self.valida_execl_path)
        no_process, right, error = 0, 0, 0
        label_dist = {}
        print_with_timestamp("-------------------验证开始---------------")
        for label_name, coin_item in res.items():
            if label_name not in label_dist:
                label_dist[label_name] = {'right': 0, 'error': 0, 'no_process': 0, 'all': 0}
            for item in coin_item:
                label_dist[label_name]["all"] += 1
                img_url = item["正面图片"]
                predict_label, predict_score = self.request_server(self.get_req_url(img_url))
                predict_name = predict_label[0]
                for it in predict_label:
                    if item["钱币名称"] in it:
                        predict_name = it
                        break
                if predict_name == "":
                    label_dist[label_name]["no_process"] += 1
                    no_process += 1
                    continue
                if label_name in predict_name:
                    label_dist[label_name]["right"] += 1
                    right += 1
                else:
                    label_dist[label_name]["error"] += 1
                    error += 1
                    print_with_timestamp(
                        label_name in predict_name,
                        label_name,
                        predict_label[:2],
                        predict_score[:2],
                        img_url,
                    )
        for finish_name, values in label_dist.items():
            print_with_timestamp(
                f"类别={finish_name}, 预测正确={values['right']}, percentage={values['right'] / (values['all'])}, no_process={values['no_process']}, 样本量={values['all']}"
            )
        print_with_timestamp(
            f"result: right={right},  all={right + error}, percentage={right / (right + error + 1)},no_process={no_process}"
        )
        print_with_timestamp("-------------------验证结束---------------")

    def request_server(self, url):
        predict_label = ["", "", "", "", ""]
        predict_score = [0, 0, 0, 0, 0]
        try:
            r = requests.get(url)
            res_js = json.loads(r.content)
            if 'data' not in res_js:
                pass
            elif self.is_local():
                predict_label = res_js["data"]
                predict_score = res_js["cls_score"]
            else:
                predict_label = res_js["others"]["coin_list"]
                predict_score = res_js["others"]["score_list"]
            return predict_label, predict_score
        except (Exception,) as e:
            print_with_timestamp(e)


if __name__ == "__main__":
    params = peOpt.parse()
    print("\n")
    print_with_timestamp(f"验证execl 输入参数: {params}")
    predict_execl = PredictExecl(peOpt.parse())
    predict_execl.valida()
