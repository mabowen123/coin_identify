import json
import os
import subprocess
import sys
import time

import requests

sys.path.append("../")
from tool import netstat, file
from image_process import load_execl
from options import predict_execl_options as peOpt
from tool.print import print_with_timestamp


class PredictExecl():
    def __init__(self, param):
        self.valida_execl_path = param.valida_execl_path
        self.model_path = param.model_path
        self.index_path = param.index_path
        self.coin_type = load_execl.get_coin_type(os.path.dirname(self.valida_execl_path))
        if not param.url and not param.model_path:
            print_with_timestamp("当--url不空时，必须提供--model_path")
            exit()
        self.index_path = param.index_path if param.index_path else os.path.join(
            os.path.dirname(self.valida_execl_path), "index.txt" if self.coin_type == 'bs' else 'index_map.txt')
        self.url = param.url
        self.classify_run_port = file.get_classify_run_port()
        self.need_run_valida_server()

    def need_run_valida_server(self):
        if not self.is_local():
            return True
        print_with_timestamp("正在启动验证服务,请稍等")
        netstat.kill_process_using_port(self.classify_run_port)
        try:
            script_directory = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_directory, "../classifyc/run.py")
            print_with_timestamp(
                f"nohup python3 {script_path} --index_path {self.index_path} --model_path {self.model_path}")
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
                r = requests.get(f'http://127.0.0.1:{self.classify_run_port}/health')
                if r.status_code == 200:
                    return True
            except (Exception,):
                pass
        print_with_timestamp("验证服务启动异常")
        exit()

    def is_local(self):
        return self.url == ""

    def get_req_url(self, image_url, image_url2=''):
        url = f"http://127.0.0.1:{self.classify_run_port}/classifyc/api/predict_old?pic1={image_url}"
        if not self.is_local() and image_url2 == '':
            url = f"http://imgt.wpt.la/ancient-coin/api/{self.url}?pic={image_url}"
        elif not self.is_local() and image_url2 != '':
            url = f"http://imgt.wpt.la/ancient-coin/api/{self.url}?pic1={image_url}&pic2={image_url2}"

        return url

    def valida(self):
        res, _ = load_execl.map_execl_to_load_image(self.valida_execl_path, self.coin_type, '版别')
        print_with_timestamp("-------------------验证开始---------------")
        if self.coin_type == 'bs':
            self.bs(res)
        elif self.coin_type == 'ns':
            self.ns(res)
        else:
            pass
        print_with_timestamp("-------------------验证结束---------------")

    def ns(self, res):
        label_dist = {}
        no_process, right, error = 0, 0, 0
        for label_name, coin_item in res.items():
            if label_name not in label_dist:
                label_dist[label_name] = {'right': 0, 'error': 0, 'no_process': 0, 'all': 0}
                for item in coin_item:
                    label_dist[label_name]["all"] += 1
                    front_img = item['正面图片']
                    back_img = item['反面图片']
                    if self.is_local():
                        predict_label_c1, predict_score1 = self.request_server(self.get_req_url(front_img))
                        predict_label_c2, predict_score2 = self.request_server(self.get_req_url(back_img))
                    else:
                        predict_label_c1, predict_score1 = self.request_server(self.get_req_url(front_img, back_img))
                        predict_label_c2, predict_score2 = ["", "", "", "", ""],[0, 0, 0, 0, 0]

                    predict_name, cls_score = self.merge_two_pic_res(predict_label_c1, predict_score1, predict_label_c2,
                                                                     predict_score2)
                    if predict_name[0] == "":
                        label_dist[label_name]["no_process"] += 1
                        no_process += 1
                        continue
                    if predict_name[0] == label_name:
                        label_dist[label_name]["right"] += 1
                        right += 1
                    else:
                        label_dist[label_name]["error"] += 1
                        error += 1
                        print(
                            fr'{predict_name[0] == label_name}【excel】标注:{label_name}【分析后】版别:{predict_name[0]} 得分:{cls_score}【背面单独】版别:{predict_label_c2[:3]} 得分:{predict_score2[:3]} 背面图:{back_img}')

        for finish_name, values in label_dist.items():
            print(
                f"类别={finish_name}, 预测正确={values['right']}, percentage={values['right'] / (values['all'])}, no_process={values['no_process']}, 样本量={values['all']}")

        print(f'result: right={right},  all={right + error}, percentage={right / (right + error)},no_process={no_process}"')


    def merge_two_pic_res(self, predict_label_ori_1, predict_score_1, predict_label_ori_2, predict_score_2,
                          threshold=0.2):
        predict_label_1 = predict_label_ori_1[0].split(';')
        predict_label_2 = predict_label_ori_2[0].split(';')
        if predict_score_2[0] < threshold and predict_score_1[0] < threshold:
            label_name = ['无法识别']
            cls_score = 0
        elif len(predict_label_1) == 1 and len(predict_label_2) == 1:
            if predict_score_1[0] > predict_score_2[0]:
                label_name = predict_label_1
                cls_score = predict_score_1[0]
            else:
                label_name = predict_label_2
                cls_score = predict_score_2[0]
        elif len(predict_label_1) == 1 and predict_score_1[0] > 0.1:
            label_name = predict_label_1
            cls_score = predict_score_1[0]
        elif len(predict_label_1) == 1 and predict_score_1[0] <= 0.1:
            label_name = predict_label_2
            cls_score = predict_score_2[0]
        elif len(predict_label_2) == 1 and predict_score_2[0] > 0.1:
            label_name = predict_label_2
            cls_score = predict_score_2[0]
        elif len(predict_label_2) == 1 and predict_score_2[0] <= 0.1:
            label_name = predict_label_1
            cls_score = predict_score_1[0]
        elif len(list(set(predict_label_2) & set(predict_label_1))) > 0:
            label_name = [it for it in predict_label_1 if it in set(predict_label_2)]
            if predict_score_1[0] > predict_score_2[0]:
                cls_score = predict_score_1[0]
            else:
                cls_score = predict_score_2[0]
        elif predict_score_2[0] > predict_score_1[0]:
            label_name = predict_label_2
            cls_score = predict_score_2[0]
        else:
            label_name = predict_label_1
            cls_score = predict_score_1[0]
        return label_name, cls_score


    def bs(self, res):
        label_dist = {}
        no_process, right, error = 0, 0, 0
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
