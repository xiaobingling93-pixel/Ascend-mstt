#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2019-2020. Huawei Technologies Co., Ltd. All rights reserved.
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
"""

import functools
import os

from inspect import isfunction
import torch
import torch.distributed as dist

from . import wrap_torch, wrap_functional, wrap_tensor, wrap_vf, wrap_distributed, wrap_aten
from .hook_module import HOOKModule
from .api_registry import api_register
from .wrap_functional import remove_dropout
from ..common.utils import check_file_or_directory_path, print_error_log, CompareException, Const, \
    print_info_log, print_warn_log, get_process_rank, torch_without_guard_version
from ..dump.utils import make_dump_dirs, DumpUtil
from ..overflow_check.utils import OverFlowUtil, clear_overflow_npu

torch_version_above_2 = torch.__version__.split('+')[0] > '2.0'

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False
    from . import wrap_npu_custom

make_dir_flag = True
REGISTER_HOOK_KWARGS = ["overflow_nums", "dump_mode", "dump_config"]


def add_clear_overflow(func, pid):
    first_module = True

    def clear_overflow_wrapper(*args, **kwargs):
        child_pid = os.getpid()
        if pid != child_pid:
            return func(*args, **kwargs)
        nonlocal first_module
        if first_module:
            clear_overflow_npu()
            first_module = False
        return func(*args, **kwargs)

    return clear_overflow_wrapper


def register_hook(model, hook, **kwargs):
    check_register_hook(hook, **kwargs)
    print_info_log("Please disable dataloader shuffle before running the program.")
    overflow_nums = kwargs.get('overflow_nums', 1)
    init_overflow_nums(overflow_nums)
    dump_mode, dump_config_file = init_dump_config(kwargs)
    if dump_mode == 'acl':
        DumpUtil.dump_switch_mode = dump_mode
        DumpUtil.set_acl_config(dump_config_file)
    if dump_mode == 'model':
        register_hook_core(hook, model)
    else:
        register_hook_core(hook)


def init_overflow_nums(overflow_nums):
    if isinstance(overflow_nums, int) and overflow_nums > 0 or overflow_nums == -1:
        OverFlowUtil.overflow_nums = overflow_nums
    else:
        print_error_log("overflow_nums must be an integer greater than 0 or set -1.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)


def check_register_hook(hook, **kwargs):
    if not isfunction(hook) or hook.__name__ not in ["overflow_check", "acc_cmp_dump"]:
        print_error_log("hook function must be set overflow_check or acc_cmp_dump")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    for item in kwargs.keys():
        if item not in REGISTER_HOOK_KWARGS:
            print_error_log(f"{item} not a valid keyword arguments in register_hook.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)


def register_hook_core(hook, model=None):
    global make_dir_flag

    pid = os.getpid()
    need_clear = True
    if make_dir_flag:
        make_dump_dirs()
        make_dir_flag = False
    hook_name = hook.__name__

    if "overflow_check" in hook_name and model is not None:
        print_error_log("Overflow check does not support model dump mode")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if "overflow_check" in hook_name and not is_gpu:
        if hasattr(torch_npu._C, "_enable_overflow_npu"):
            torch_npu._C._enable_overflow_npu()
            print_info_log("Enable overflow function success.")
        else:
            print_warn_log("Api '_enable_overflow_npu' is not exist, "
                           "the overflow detection function on milan platform maybe not work! "
                           "please check the version of software torch_npu.")
        # In NPU scene, clear the overflow flag before overflow detection
        if need_clear:
            HOOKModule.__init__ = add_clear_overflow(HOOKModule.__init__, pid)

    print_info_log("Start mounting the {} hook function to the model.".format(hook_name))
    hook = functools.partial(hook, dump_step=0, pid=pid)
    print_info_log("The {} hook function is successfully mounted to the model.".format(hook_name))

    if model is not None:
        if not isinstance(model, torch.nn.Module):
            print_error_log("The argument model must be an object of torch.nn.Module")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
        for _, module in model.named_modules():
            if "torch.nn.modules" in str(module.__class__):
                prefix = "Module_" + module.__class__.__name__
                module.register_forward_hook(hook(prefix + "_{}_" + "forward"))
                module.register_backward_hook(hook(prefix + "_{}_" + "backward"))
    else:
        api_register.initialize_hook(hook)
        api_register.api_modularity()

    if "acc_cmp_dump" in hook_name:
        remove_dropout()


def init_dump_config(kwargs):
    dump_mode = kwargs.get('dump_mode', "api")
    dump_config = kwargs.get('dump_config')
    dump_config_file = ''
    if dump_mode not in Const.SUPPORT_DUMP_MODE:
        print_error_log("dump_mode only support %s" % Const.SUPPORT_DUMP_MODE)
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if dump_mode == "acl":
        if dump_config is None:
            print_error_log("dump_mode is acl mode, dump_config must be configured.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
        dump_config_file = os.path.realpath(dump_config)
        check_file_or_directory_path(dump_config_file)
        if not dump_config.endswith(".json"):
            print_error_log("dump_config must be configure json file.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
    return dump_mode, dump_config_file
