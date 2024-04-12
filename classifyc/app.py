from collections import OrderedDict

from flask import Flask
# from flask_limiter import Limiter
from werkzeug.middleware.dispatcher import DispatcherMiddleware

import config, log
import time
from views import json_api

# from dotenv import load_dotenv, find_dotenv
# import os, sys
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # 添加系统环境变量
# sys.path.append(BASE_DIR)

# load_dotenv(find_dotenv())

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, OrderedDict((
        (config.API_PATH, json_api),
    )))
    return app


flask_app = create_app()
# limiter = Limiter(
#     flask_app,
#     key_func=lambda: '0.0.0.0',
#     default_limits=[config.DEFAULT_LIMITS]
# )


def bootstrap(app):
    log.setup(json_api.logger)
    log.setup(app.logger)


@flask_app.route(f'/health', methods=['GET'])
def health():
    return {'code': 0, 'msg': 'ok', 'nowTime': int(time.time())}
