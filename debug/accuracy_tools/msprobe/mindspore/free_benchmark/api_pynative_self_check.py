# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import functools
import importlib
import os
import threading
import traceback

import mindspore as ms

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.core.common.file_utils import check_path_length, load_yaml
from msprobe.core.common.runtime import Runtime
from msprobe.core.hook_manager import HookSet
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.common.utils import get_rank_if_initialized
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.hook_cell.api_register import get_api_register
from msprobe.mindspore.dump.hook_cell.hook_cell import HOOKCell
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.common.handler_params import HandlerParams
from msprobe.mindspore.free_benchmark.common.utils import Tools
from msprobe.mindspore.free_benchmark.handler.handler_factory import HandlerFactory
from msprobe.mindspore.free_benchmark.perturbation.perturbation_factory import PerturbationFactory

_api_register = get_api_register()


class ApiPyNativeSelfCheck:
    def __init__(self, config: DebuggerConfig):
        Config.is_enable = True
        Config.handler_type = config.handler_type
        Config.pert_type = config.pert_type
        Config.stage = config.stage
        Config.dump_level = config.dump_level
        Config.steps = config.step
        Config.ranks = config.rank
        Config.dump_path = os.path.join(config.dump_path, FreeBenchmarkConst.CHECK_RESULT_FILE)
        check_path_length(Config.dump_path)

        self.ori_func = {}

        self.api_list = config.list
        all_api = get_supported_ops()
        if not self.api_list:
            self.api_list = all_api
        else:
            self.api_list = set(self.api_list) & all_api
        self.store_original_func()

    def handle(self):
        _api_register.initialize_hook(self.build_hook)
        _api_register.register_all_api()

    def build_hook(self, api_name):
        def pre_hook(cell, input_data):
            return None

        def forward_hook(api_name_with_id, cell, input_data, output_data):
            ret = None
            tid = threading.get_ident()

            if not need_wrapper_func():
                del cell.msprobe_input_kwargs[tid]
                return ret

            api_name_with_id = api_name_with_id[:-1]
            hook_prefix = api_name_with_id[:api_name_with_id.find(Const.SEP) + 1]
            api_name = (MsConst.HOOK_MS_PREFIX_DICT.get(hook_prefix, "") +
                        api_name_with_id[api_name_with_id.find(Const.SEP) + 1:api_name_with_id.rfind(Const.SEP)])
            if api_name in self.api_list:
                ret = check_self(api_name_with_id, output_data, self.ori_func.get(api_name),
                                 *input_data, **cell.msprobe_input_kwargs[tid])

            del cell.msprobe_input_kwargs[tid]
            return ret

        def backward_hook(cell, grad_input, grad_output):
            pass

        HOOKCell.get_cell_count(api_name)
        api_name_with_id = api_name + str(HOOKCell.get_cell_count(api_name)) + Const.SEP
        forward_hook = functools.partial(forward_hook, api_name_with_id)
        HOOKCell.add_cell_count(api_name)

        def wrap_forward_hook(cell, input_data, output_data):
            return forward_hook(cell, input_data, output_data)

        def wrap_backward_hook(cell, grad_input, grad_output):
            return backward_hook(cell, grad_input, grad_output)

        def pre_backward_hook(cell, grad_input):
            return None
        
        return HookSet(
            forward_hook=wrap_forward_hook,
            forward_pre_hook=pre_hook,
            backward_hook=wrap_backward_hook,
            backward_pre_hook=pre_backward_hook
        )

    def store_original_func(self):
        for api_name in self.api_list:
            self.ori_func[api_name] = get_module(api_name)[1]


def get_supported_ops():
    supported_ops = []
    cur_path = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(cur_path, "data", FreeBenchmarkConst.SUPPORTED_CHECK_API_FILE)

    supported_ops_list = load_yaml(yaml_path)
    for k, v in FreeBenchmarkConst.API_PREFIX_DICT.items():
        ops = supported_ops_list.get(k)
        if ops:
            ops = [v + i for i in ops]
            supported_ops += ops

    _all_functional_ops = []
    ms_ops = dir(ms.ops)
    ms_ops = [MsConst.OPS_PREFIX + i for i in ms_ops]
    _all_functional_ops += ms_ops

    ms_tensor = dir(ms.Tensor)
    ms_tensor = [MsConst.TENSOR_PREFIX + i for i in ms_tensor]
    _all_functional_ops += ms_tensor

    ms_mint = dir(ms.mint)
    ms_mint = [MsConst.MINT_PREFIX + i for i in ms_mint]
    _all_functional_ops += ms_mint

    ms_mint_nn_func = dir(ms.mint.nn.functional)
    ms_mint_nn_func = [MsConst.MINT_NN_FUNC_PREFIX + i for i in ms_mint_nn_func]
    _all_functional_ops += ms_mint_nn_func

    return set(supported_ops) & set(_all_functional_ops)


def get_module(api_name):
    func_name_list = api_name.split(Const.SEP)
    func_name = func_name_list[-1]
    module_obj = importlib.import_module(func_name_list[0])
    for i, module_name in enumerate(func_name_list[1:-1]):
        if not hasattr(module_obj, module_name):
            importlib.import_module(f"{Const.SEP.join(func_name_list[:i + 2])}")
        module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    return module_obj, orig_func


def check_self(api_name_with_id, output, ori_func, *args, **kwargs):
    ret = None

    if Config.stage == Const.BACKWARD and not (check_all_tensor(args) and check_all_tensor(output)):
        logger.warning(f"{api_name_with_id} has non-tensor input or output.")
        return ret

    params = data_pre_deal(api_name_with_id, ori_func, *args, **kwargs)
    if params.index == -1:
        return ret

    logger.info(f"[{api_name_with_id}] is {Config.handler_type}ing.")
    _api_register.restore_all_api()

    try:
        perturbation = PerturbationFactory.create(api_name_with_id)
        params.fuzzed_result = perturbation.handle(params)
        if params.fuzzed_result is False:
            _api_register.register_all_api()
            return ret
        if Config.stage == Const.BACKWARD:
            params.original_result = Tools.get_grad(params.original_func, *params.args, **params.kwargs)
        else:
            params.original_result = output
        ret = deal_fuzzed_and_original_result(api_name_with_id, params)
    except Exception as e:
        logger.error(f"[{api_name_with_id}] Error: {str(e)}")
        logger.error(f"[{api_name_with_id}] Error detail: {traceback.format_exc()}")

    _api_register.register_all_api()
    return ret


def check_all_tensor(input_output):
    if isinstance(input_output, ms.Tensor):
        return True
    if isinstance(input_output, (tuple, list)):
        return all([check_all_tensor(v) for v in input_output])
    return False


def get_target_arg_index(args) -> int:
    """
    类型校验

    """
    for i, arg in enumerate(args):
        if ms.ops.is_tensor(arg):
            if not ms.ops.is_floating_point(arg):
                continue
            return i
        if isinstance(arg, (list, tuple, dict)):
            return i
    return -1


def data_pre_deal(api_name_with_id, func, *args, **kwargs):
    params = HandlerParams()
    params.args = args
    params.kwargs = kwargs
    params.original_func = func
    index = get_target_arg_index(args)
    if index == -1:
        logger.warning(f"{api_name_with_id} has no supported input type.")
    params.index = index
    return params


def need_wrapper_func():
    if not (Runtime.is_running and Config.is_enable):
        return False

    if Config.steps and Runtime.step_count not in Config.steps:
        return False

    if Runtime.rank_id == -1:
        try:
            Runtime.rank_id = get_rank_if_initialized()
        except DistributedNotInitializedError:
            Runtime.rank_id = -1
    if Config.ranks and Runtime.rank_id != -1 and Runtime.rank_id not in Config.ranks:
        return False

    return True


def deal_fuzzed_and_original_result(api_name_with_id, params: HandlerParams):
    handler = HandlerFactory.create(api_name_with_id)
    result = handler.handle(params)
    return result
