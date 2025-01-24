# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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
