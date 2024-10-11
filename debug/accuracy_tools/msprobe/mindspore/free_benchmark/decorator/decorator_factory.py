# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

import os
import sys
import traceback
from functools import wraps
from typing import Dict, List, Tuple

from mindspore import ops

from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.decorator.dec_forward import ForwardSelfChecker
from msprobe.mindspore.runtime import Runtime


def decorate(original_func, decorate_func, api_name=None):
    """
    总装饰器
    """
    @wraps(original_func)
    def fuzz_wrapper(*args, **kwargs):

        def __exec_decorate_func():
            params = data_pre_deal(api_name, original_func, *args, **kwargs)
            result = decorate_func(params)
            return result

        try:
            if Runtime.rank_id == -1:
                Runtime.rank_id = os.environ.get("RANK_ID", -1)
            if need_wrapper_func():
                logger.info(f"[{api_name}] is checking.")
                return __exec_decorate_func()
        except Exception as e:
            logger.error(f"[{api_name}] Error: {str(e)}")
            logger.error(f"[{api_name}] Error detail: {traceback.format_exc()}")

        return original_func(*args, **kwargs)

    return fuzz_wrapper


def decorate_forward_function(func, api_name=None):
    """
    前向装饰器
    """

    if not api_name:
        api_name = func.__name__

    def forward_func(params: HandlerParams):
        forward = ForwardSelfChecker(api_name)
        result = forward.handle(params)
        return result

    return decorate(func, forward_func, api_name)


def stack_depth_check() -> bool:
    nested_depth = 1
    frame = sys._getframe(1)
    while frame:
        if frame.f_code.co_name == "fuzz_wrapper":
            nested_depth -= 1
            if nested_depth < 0:
                return False
        frame = frame.f_back
    return True


def get_target_arg_index(args: Tuple) -> int:
    """
    类型校验

    """
    for i, arg in enumerate(args):
        if ops.is_tensor(arg):
            if not ops.is_floating_point(arg):
                continue
            return i
        if isinstance(arg, (List, Tuple, Dict)):
            return i
    return -1


def data_pre_deal(api_name, func, *args, **kwargs):
    params = HandlerParams()
    params.args = args
    params.kwargs = kwargs
    params.original_func = func
    index = get_target_arg_index(args)
    if index == -1:
        raise Exception(f"{api_name} has no supported input type")
    params.index = index
    return params


def need_wrapper_func():
    if not (Runtime.is_running and Config.is_enable):
        return False
    if not stack_depth_check():
        return False
    if Config.steps and Runtime.step_count not in Config.steps:
        return False
    if Config.ranks and Runtime.rank_id != -1 and Runtime.rank_id not in Config.ranks:
        return False
    return True
