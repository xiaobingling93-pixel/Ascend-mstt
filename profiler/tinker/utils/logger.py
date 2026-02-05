# -*- coding: utf-8 -*-

# Copyright (c) 2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" 日志模块
"""
import logging.config
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional


class CustomFormatter(logging.Formatter):

    def format(self, record):
        log_msg = ""
        # 当打印日志级别高于 INFO，各进程都要输出日志，且格式中带rank
        if record.levelno > logging.INFO:
            current_time = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
            log_msg += f"{current_time} "

            gpu_rank = torch.distributed.get_rank()
            log_msg += f"[NPU Rank: {gpu_rank}] "

            file_name = record.pathname
            func_name = record.funcName
            line_no = record.lineno
            log_msg += f"{file_name} - {func_name}():{line_no} - "

            log_msg += f"{record.levelname}: "

            log_msg += f"{record.msg}"
            return log_msg

        # 打印日志级别不高于 INFO，直出msg
        if torch.distributed.get_rank() == 0:
            log_msg += f"{record.msg}"
            return log_msg
        return None


class CustomHandler(logging.StreamHandler):
    def emit(self, record):
        log_entry = self.format(record)
        if log_entry:  # 只有当 log_entry 不为空时才输出
            self.stream.write(log_entry + self.terminator)


class Formatters(Enum):
    DEFAULT = {
        'standard_formatter': {
            'format': '{message}',
            'style': '{',
        },
        'verbose_formatter': {
            'format': '{message}',
            'style': '{',
        }
    }
    PROFILE = {
        'standard_formatter': {
            '()': 'tinker.utils.logger.CustomFormatter'
        },
        'verbose_formatter': {
            '()': 'tinker.utils.logger.CustomFormatter'
        }
    }


# 建两个，仅避免频繁切换此让日志为默认使用
logger = logging.getLogger('default')

# 在多进程环境下使用
profile_logger = logging.getLogger('profile')


def get_default_config():
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard_formatter': {},
            'verbose_formatter': {}
        },
        'handlers': {
            # 控制台输出
            'console': {
                # init 更新
                'level': 'NOTSET',
                'class': 'logging.StreamHandler',
                'formatter': 'standard_formatter'  # 输出格式
            },
            # 日志输出
            'logfile': {
                # init 更新
                'level': 'NOTSET',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': '',
                'mode': 'a',
                'maxBytes': 1024 * 1024 * 100,  # 文件大小
                'backupCount': 5,  # 备份数
                'formatter': 'verbose_formatter',  # 输出格式
                'encoding': 'utf-8',  # 设置默认编码
            }
        },
        # 配置用哪几种 handlers 来处理日志
        'loggers': {
            # log 调用时需要当作参数传入
        }
    }
    return log_config


def init_log(log_path: Optional[str], log_level):
    """
    此方法用于初始化&更新默认的log
    :param log_path: 日志存的地址
    :param log_level: 等级
    :return:
    """
    log_config = get_default_config()
    formatter = Formatters.DEFAULT

    # 更新 loggers：
    log_config['loggers']['default'] = {
        'handlers': ['console'],
        'level': 'NOTSET',
        'propagate': False
    }

    # 更新 level
    log_config['handlers']['console']['level'] = log_level
    log_config['handlers']['logfile']['level'] = log_level
    log_config['loggers']['default']['level'] = log_level

    # 更新 formatter
    log_config['formatters'].update(formatter.value)

    # 写入文件不一定需要有
    if log_path:
        log_config['loggers']['default']['handlers'].append('logfile')
        # 更新 file_name
        log_config['handlers']['logfile']['filename'] = log_path
    else:
        del log_config['handlers']['logfile']
    logging.config.dictConfig(log_config)


def init_profile_log(log_level):
    """
    在多进程环境中使用的log，不支持文件输出
    :param log_level: 等级
    :return:
    """
    log_config = get_default_config()
    formatter = Formatters.PROFILE

    # 更新 loggers：
    log_config['loggers']['profile'] = {
        'handlers': ['console'],
        'level': 'NOTSET',
        'propagate': False
    }
    import torch
    global torch
    if not torch.distributed.is_initialized():
        raise RuntimeError('Before using method \'init_profile_log\', torch should be initialized.')
        # 输出到 console

    # 更新 level
    log_config['handlers']['console']['level'] = log_level
    log_config['handlers']['logfile']['level'] = log_level
    log_config['loggers']['profile']['level'] = log_level

    # 更新handler，此处handler对logger打印的空字符串，默认不打印
    log_config['handlers']['console']['class'] = "tinker.utils.logger.CustomHandler"

    # 更新 formatter
    log_config['formatters'].update(formatter.value)

    # 删除输出文件
    del log_config['handlers']['logfile']
    logging.config.dictConfig(log_config)