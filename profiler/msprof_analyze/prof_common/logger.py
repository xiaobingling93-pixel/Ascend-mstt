# -------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import logging
import os

from msprof_analyze.prof_common.constant import Constant


def get_log_level():
    log_level = os.getenv(Constant.MSPROF_ANALYZE_LOG_LEVEL, Constant.DEFAULT_LOG_LEVEL).upper()
    if not hasattr(logging, log_level):
        raise AttributeError(f"module 'logging' has no attribute '{log_level}', "
                             f"supported log level: {', '.join(Constant.SUPPORTED_LOG_LEVEL)}")
    return log_level


def get_logger() -> logging.Logger:
    logger_name = "msprof-analyze"
    if logger_name in logging.Logger.manager.loggerDict:
        return logging.getLogger(logger_name)

    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(get_log_level())

    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
