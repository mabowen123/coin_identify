# -*- coding: utf-8 -*-
# @Time    : 2019/12/27 15:44
# @Author  : William Wang
# @File    : log.py

# 初始化日志

import os
import re
import json
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from flask import Flask, has_request_context, request, logging as log

from classifyc import config


DEPTARTMENT_NAME = 'da'
BASE_DIR = '../log'
LOGFILE = f"info/{DEPTARTMENT_NAME}-{config.APP_NAME}.log"
ERRORLOGFILE = f"error/{DEPTARTMENT_NAME}-{config.APP_NAME}.log"
ENV = os.environ.get('NAMESPACE', '')
NODE_NAME = os.environ.get('NODE_NAME', '')


EXT_MATCH = {
    ".%Y-%m-%d_%H-%M-%S": re.compile(r"^.\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(\.\w+)?$", re.ASCII),
    ".%Y-%m-%d_%H-%M": re.compile(r"^.\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(\.\w+)?$", re.ASCII),
    ".%Y-%m-%d_%H": re.compile(r"^.\d{4}-\d{2}-\d{2}_\d{2}(\.\w+)?$", re.ASCII),
    ".%Y-%m-%d": re.compile(r"^.\d{4}-\d{2}-\d{2}(\.\w+)?$", re.ASCII),
}


def namer(name):
    path, filename = os.path.split(name)
    base_name, date_time = os.path.splitext(filename)
    base_name, suffix = os.path.splitext(base_name)
    for time_str, rr in EXT_MATCH.items():
        if rr.match(date_time):
            local_datetime = datetime.strptime(date_time, time_str)
            date_time = local_datetime.strftime('.%Y%m%d%H%M')
    return os.path.join(path, base_name + date_time + suffix)


class MyTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, *args, **kwargs):
        super(MyTimedRotatingFileHandler, self).__init__(*args, **kwargs)

    def getFilesToDelete(self):
        """
        Determine the files to delete when rolling over.

        More specific than the earlier method, which just used glob.glob().
        """
        dirName, baseNameFull = os.path.split(self.baseFilename)
        baseName = os.path.splitext(baseNameFull)[0]
        fileNames = os.listdir(dirName)
        result = []
        prefix = baseName + "."
        plen = len(prefix)
        for fileName in fileNames:
            if fileName[:plen] == prefix:
                # suffix = fileName[plen:]
                # if self.extMatch.match(suffix):
                result.append(os.path.join(dirName, fileName))
        if len(result) < self.backupCount:
            result = []
        else:
            result.sort()
            result = result[:len(result) - self.backupCount]
        return result


class CommonFormatter(logging.Formatter):
    def_keys = ['name', 'msg', 'args', 'levelname', 'levelno',
                'pathname', 'filename', 'module', 'exc_info',
                'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread',
                'threadName', 'processName', 'process', 'message',
                'project', 'url', 'remote_addr', 'stack']

    def format(self, record):
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
        else:
            record.url = None
            record.remote_addr = None
        record.stack = ""
        if record.stack_info:
            record.stack = str(record.stack_info).split("\n")

        record.stack_info = ""
        extra = {k: v for k, v in record.__dict__.items()
                 if k not in self.def_keys}
        log_cont = super().format(record)
        if len(extra) > 0:
            new_extra = {}
            for k, v in extra.items():
                if isinstance(v, (dict, list, tuple)):
                    new_extra[k] = json.dumps(v, ensure_ascii=False)
                else:
                    new_extra[k] = v
            new_log_count = json.loads(log_cont)
            new_log_count["fields"].update(new_extra)
            # extra.update()
            log_cont = json.dumps(new_log_count, ensure_ascii=False)

        return log_cont


class ContextFilter(logging.Filter):
    """Enhances log messages with contextual information"""

    def filter(self, record):
        if record.levelno == logging.INFO or record.levelno == logging.WARNING:
            return True

        return False


def setup(logger: Flask.logger):
    logging.addLevelName(logging.DEBUG, "debug")
    logging.addLevelName(logging.INFO, "info")
    logging.addLevelName(logging.ERROR, "error")
    logging.addLevelName(logging.WARN, "warning")
    logging.addLevelName(logging.WARNING, "warning")

    formatter = CommonFormatter(
        '{"timestamp":"%(asctime)s.%(msecs)03d+08:00",'
        '"level":"%(levelname)s",'
        '"project_name":"' + DEPTARTMENT_NAME + "-" + config.APP_NAME + '",'
        '"msg": "%(message)s",'
        '"file_line": "%(filename)s:%(lineno)s",'
        '"env": "' + ENV + '",'
        '"node_name": "' + NODE_NAME + '",'
        '"fields":{"remote_addr": "%(remote_addr)s", "url": "%(url)s"}'
        '}',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    info_log_path = os.path.join(BASE_DIR, LOGFILE)
    err_log_path = os.path.join(BASE_DIR, ERRORLOGFILE)
    info_path, _ = os.path.split(info_log_path)
    err_path, _ = os.path.split(err_log_path)
    if not os.path.exists(info_path):
        os.makedirs(info_path)
    if not os.path.exists(err_path):
        os.makedirs(err_path)

    handler = MyTimedRotatingFileHandler(
        info_log_path, when="D", interval=1, backupCount=15,
        encoding="UTF-8", delay=False, utc=False)
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    handler.namer = namer

    # 在main函数之中， 增加加载这个filter即可
    handler.addFilter(ContextFilter())

    logger.addHandler(handler)

    err_handler = MyTimedRotatingFileHandler(
        err_log_path, when="D", interval=1, backupCount=15,
        encoding="UTF-8", delay=False, utc=False)
    err_handler.setFormatter(formatter)
    err_handler.setLevel(logging.ERROR)
    err_handler.namer = namer
    logger.addHandler(err_handler)
    logger.setLevel(logging.INFO)

    logger.removeHandler(log.default_handler)
