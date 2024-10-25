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

import importlib
import inspect
import os

import mindspore as ms
from mindspore.communication import comm_func

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import check_path_length, load_yaml
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.decorator.decorator_factory import decorate_forward_function


class ApiPyNativeSelFCheck:
    def __init__(self, config: DebuggerConfig):
        Config.is_enable = True
        Config.handler_type = config.handler_type
        Config.pert_type = config.pert_type
        Config.stage = config.stage
        Config.dump_level = config.dump_level
        Config.steps = config.step
        Config.ranks = config.rank
        Config.dump_path = os.path.join(config.dump_path, "free_benchmark.csv")
        check_path_length(Config.dump_path)

        self.api_list = config.list
        all_api = get_supported_ops()
        if not self.api_list:
            self.api_list = all_api
        else:
            self.api_list = set(self.api_list) & all_api

    def handle(self):
        for api_name in self.api_list:
            hijack(api_name)


def get_supported_ops():
    supported_ops = []
    cur_path = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(cur_path, "data", "support_wrap_ops.yaml")

    yaml_data = load_yaml(yaml_path)
    for k, v in FreeBenchmarkConst.API_PREFIX_DICT.items():
        ops = yaml_data.get(k)
        if ops:
            ops = [v + i for i in ops]
            supported_ops += ops

    _all_functional_ops = []
    ms_ops = dir(ms.ops)
    ms_ops = [MsConst.OPS_PREFIX + i for i in ms_ops]
    _all_functional_ops += ms_ops

    ms_tensor = dir(ms.Tensor)
    ms_tensor = [MsConst.Tensor_PREFIX + i for i in ms_tensor]
    _all_functional_ops += ms_tensor

    ms_mint = dir(ms.mint)
    ms_mint = [MsConst.MINT_PREFIX + i for i in ms_mint]
    _all_functional_ops += ms_mint

    ms_mint_nn_func = dir(ms.mint.nn.functional)
    ms_mint_nn_func = [MsConst.MINT_NN_FUNC_PREFIX + i for i in ms_mint_nn_func]
    _all_functional_ops += ms_mint_nn_func

    ms_communication = dir(comm_func)
    ms_communication = [MsConst.COMM_PREFIX + i for i in ms_communication]
    _all_functional_ops += ms_communication

    return set(supported_ops) & set(_all_functional_ops)


def get_decorate_func():
    return decorate_forward_function


def is_func_support_decorate(orig_func):
    return not inspect.isclass(orig_func) and callable(orig_func)


def get_wrapper_obj(orig_func, api_name):
    if is_func_support_decorate(orig_func):
        wrapped_obj = get_decorate_func()(orig_func, api_name)
    else:
        wrapped_obj = orig_func
    return wrapped_obj


def get_module(api_name):
    func_name_list = api_name.split(Const.SEP)
    func_name = func_name_list[-1]
    module_obj = importlib.import_module(func_name_list[0])
    for i, module_name in enumerate(func_name_list[1:-1]):
        if not hasattr(module_obj, module_name):
            importlib.import_module(f"{Const.SEP.join(func_name_list[:i+2])}")
        module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    return module_obj, orig_func


def hijack(api_name):
    if not api_name.strip():
        return
    try:
        func_name = api_name.split(Const.SEP)[-1]
        module_obj, origin_func = get_module(api_name)
        wrapped_obj = get_wrapper_obj(origin_func, api_name)
        setattr(module_obj, func_name, wrapped_obj)
    except Exception as e:
        logger.error(f"Failed decorator {api_name}: {e}")
