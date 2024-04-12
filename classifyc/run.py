from gevent import monkey

monkey.patch_all(thread=False)



from gevent import pywsgi
from app import flask_app as app
from app import bootstrap

bootstrap(app)


def run_forever():
    server = pywsgi.WSGIServer(('0.0.0.0', 86), app)
    server.serve_forever()


if __name__ == "__main__":
    run_forever()
