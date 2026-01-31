# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import logging

from msprobe.core.common.exceptions import FileCheckException


class ParseException(Exception):
    PARSE_INVALID_PATH_ERROR = 0
    PARSE_NO_FILE_ERROR = 1
    PARSE_NO_MODULE_ERROR = 2
    PARSE_INVALID_DATA_ERROR = 3
    PARSE_INVALID_FILE_FORMAT_ERROR = 4
    PARSE_UNICODE_ERROR = 5
    PARSE_JSONDECODE_ERROR = 6
    PARSE_MSACCUCMP_ERROR = 7
    PARSE_LOAD_NPY_ERROR = 8
    PARSE_INVALID_PARAM_ERROR = 9

    def __init__(self, code, error_info=""):
        super(ParseException, self).__init__()
        self.error_info = error_info
        self.code = code


def catch_exception(func):
    def inner(*args, **kwargs):
        log = logging.getLogger()
        line = args[-1] if len(args) == 2 else ""
        result = None
        try:
            result = func(*args, **kwargs)
        except OSError:
            log.error("%s: command not found" % line)
        except ParseException:
            log.error("Command execution failed")
        except FileCheckException:
            log.error("Command execution failed")
        return result

    return inner
