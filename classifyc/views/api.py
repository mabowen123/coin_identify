import requests
from flask import Flask, request
import cv2
import numpy as np

from classifyc import config
from classifyc.views.predict import predict_two_pics, predict_old_feature, detect_coin_old_bei

from classifyc.views.utils import ApiResult
from classifyc.views.exceptions import APIException, RequestIOError, ParamsValidationError
import time


# from flask_limiter import Limiter
# from flask_limiter.util import get_remote_address


class FlaskAPI(Flask):
    def make_response(self, rv):
        if isinstance(rv, dict):
            if 'code' not in rv:
                rv['code'] = 0
            if 'nowTime' not in rv:
                rv['nowTime'] = int(time.time())
            rv = ApiResult(rv)
        if isinstance(rv, ApiResult):
            return rv.to_response()
        return Flask.make_response(self, rv)


def create_app():
    app = FlaskAPI(__name__)
    app.config.from_object(config)
    return app


json_api = create_app()


# limiter = Limiter(
#     json_api,
#     key_func=get_remote_address,   # 根据访问者的IP记录访问次数
#     default_limits=[config.DEFAULT_LIMITS]  # 默认限制，一天最多访问200次，一小时最多访问50次
# )


@json_api.errorhandler(APIException)
def api_error_handler(error: APIException):
    from flask import current_app
    if error.status != 400:
        current_app.logger.error(error.message.replace("'", ' ').replace('"', ' ').replace('\n', ''), stack_info=False)
    return error.to_result()


@json_api.route('/predict', methods=['GET', 'POST'])
# @limiter.limit(config.DEFAULT_LIMITS)
def image_truncate():
    res = ""
    detect_crop_xy = []
    cls_score = 0
    # pic: Picture = unmarshal_json(PictureSchema)
    try:
        image_url = request.args.get("pic1")
        image_url_2 = request.args.get("pic2")
        if image_url is '' or image_url_2 is '':
            raise ParamsValidationError(400, "no args pic url", 400)
        image_bytes = requests.get(image_url, timeout=10).content
        image_pil_1 = cv2.imdecode(np.fromstring(image_bytes, np.uint8), 1)
        image_bytes_2 = requests.get(image_url_2, timeout=10).content
        image_pil_2 = cv2.imdecode(np.fromstring(image_bytes_2, np.uint8), 1)
        if image_pil_1 is None or image_pil_2 is None:
            raise ParamsValidationError(400, str('Image url is invalid, please check'), 400)
    except Exception as e:
        raise ParamsValidationError(400, str(e), 400)

    try:
        res, detect_crop_xy, cls_score = predict_two_pics(image_pil_1, image_pil_2)
        return {'code': 0, 'data': res, 'msg': '', 'detect_crop_xy': detect_crop_xy, 'cls_score': cls_score}
    except Exception as e:
        raise RequestIOError(500, str(e), 500)
    finally:
        json_api.logger.info("classify_coin predict process down : ",
                             extra={'img_url': image_url, 'data': res, 'detect_crop_xy': detect_crop_xy,
                                    'cls_score': cls_score})


@json_api.route('/predict_old', methods=['GET', 'POST'])
# @limiter.limit(config.DEFAULT_LIMITS)
def image_truncate_old():
    res = ""
    detect_crop_xy = []
    cls_score = 0
    # pic: Picture = unmarshal_json(PictureSchema)
    try:
        image_url = request.args.get("pic1")
        # print(image_url)
        if image_url is '':
            raise ParamsValidationError(400, "no args pic url", 400)
        image_bytes = requests.get(image_url, timeout=10).content
        image_pil_1 = cv2.imdecode(np.fromstring(image_bytes, np.uint8), 1)
        if image_pil_1 is None:
            raise ParamsValidationError(400, str('Image url is invalid, please check'), 400)
    except Exception as e:
        raise ParamsValidationError(400, str(e), 400)

    try:
        res, detect_crop_xy, cls_score = predict_old_feature(image_pil_1)
        # res, detect_crop_xy, cls_score = detect_coin_old_bei(image_pil_1)
        return {'code': 0, 'data': res, 'msg': '', 'detect_crop_xy': detect_crop_xy, 'cls_score': cls_score}
    except Exception as e:
        raise RequestIOError(500, str(e), 500)
    finally:
        json_api.logger.info("classify_coin predict process down : ",
                             extra={'img_url': image_url, 'data': res, 'detect_crop_xy': detect_crop_xy,
                                    'cls_score': cls_score})


@json_api.route('/inference', methods=['POST'])
# @limiter.limit(config.DEFAULT_LIMITS)
def image_inference():
    res = ""
    detect_crop_xy = []
    cls_score = 0
    start_time = time.time()
    try:
        image_bytes = request.files['img1'].read()
        image_bytes2 = request.files['img2'].read()
    except Exception as e:
        raise ParamsValidationError(400, str('Image bytes is invalid, please check'), 400)

    try:
        image_pil_1 = cv2.imdecode(np.fromstring(image_bytes, np.uint8), 1)
        image_pil_2 = cv2.imdecode(np.fromstring(image_bytes2, np.uint8), 1)

        if image_pil_1 is None or image_pil_2 is None:
            raise ParamsValidationError(400, str('Image url is invalid, please check'), 400)
    except Exception as e:
        raise ParamsValidationError(400, str(e), 400)

    try:
        res, detect_crop_xy, cls_score = predict_two_pics(image_pil_1, image_pil_2)
        return {'data': res, 'detect_crop_xy': detect_crop_xy, 'cls_score': cls_score}

    except Exception as e:
        raise RequestIOError(500, str(e), 500)

    finally:
        json_api.logger.info("classify_coin inference process down : ",
                             extra={'cost_time': (time.time() - start_time),
                                    'data': res, 'detect_crop_xy': detect_crop_xy, 'cls_score': cls_score})
