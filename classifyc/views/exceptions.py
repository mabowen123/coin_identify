from classifyc.views.utils import ApiResult
import time
from classifyc import config


class APIException(Exception):

    def __init__(self, status, message, code=0, raw_message=None):
        self.message = message
        self.status = status
        self.code = code
        self.raw_message = raw_message or message

    def to_result(self):
        return ApiResult({'msg': config.APP_NAME + ' service occur error: ' + self.message, 'code': self.code,
                          'nowTime': int(time.time())}, status=self.status)


class ParamsValidationError(APIException):
    pass


class RequestIOError(APIException):
    pass
