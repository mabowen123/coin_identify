from gevent import monkey
import sys

sys.path.append("../")
from tool import file

monkey.patch_all(thread=False)

from gevent import pywsgi
from app import flask_app as app
from app import bootstrap

bootstrap(app)


def run_forever():
    classify_run_port = file.get_classify_run_port()
    server = pywsgi.WSGIServer(('0.0.0.0', classify_run_port), app)
    server.serve_forever()


if __name__ == "__main__":
    run_forever()
