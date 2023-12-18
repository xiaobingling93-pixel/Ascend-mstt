#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2023-2023. Huawei Technologies Co., Ltd. All rights reserved.
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

from api_accuracy_checker.dump.api_info import ForwardAPIInfo, BackwardAPIInfo
from api_accuracy_checker.dump.info_dump import write_api_info_json, initialize_output_json
from api_accuracy_checker.common.utils import print_error_log, CompareException
from api_accuracy_checker.hook_module.register_hook import initialize_hook
from api_accuracy_checker.common.config import msCheckerConfig


def set_dump_switch(switch):
    if switch not in ["ON", "OFF"]:
        print_error_log("Please set switch with 'ON' or 'OFF'.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if switch == "ON":
        initialize_hook(pretest_hook)
        initialize_output_json()
    DumpUtil.set_dump_switch(switch)


def start():
    if not DumpUtil.get_dump_switch() and not msCheckerConfig.enable_dataloader:
        DumpUtil.incr_iter_num_maybe_exit()


def stop():
    DumpUtil.set_dump_switch("OFF")


def step():
    if not msCheckerConfig.enable_dataloader:
        DumpUtil.call_num += 1


class DumpUtil(object):
    dump_switch = None
    call_num = 0

    @staticmethod
    def set_dump_switch(switch):
        DumpUtil.dump_switch = switch

    @staticmethod
    def get_dump_switch():
        return DumpUtil.dump_switch == "ON"

    @staticmethod
    def incr_iter_num_maybe_exit():
        if DumpUtil.call_num in msCheckerConfig.target_iter:
            set_dump_switch("ON")
        elif DumpUtil.call_num > max(msCheckerConfig.target_iter):
            raise Exception("Model pretest: exit after iteration {}".format(DumpUtil.call_num - 1))
        else:
            set_dump_switch("OFF")


class DumpConst:
    delimiter = '*'
    forward = 'forward'
    backward = 'backward'


def pretest_info_dump(name, out_feat, module, phase):
    if not DumpUtil.get_dump_switch():
        return
    if phase == DumpConst.forward:
        api_info = ForwardAPIInfo(name, module.input_args, module.input_kwargs)
    elif phase == DumpConst.backward:
        api_info = BackwardAPIInfo(name, out_feat)
    else:
        msg = "Unexpected training phase {}.".format(phase)
        print_error_log(msg)
        raise NotImplementedError(msg)

    write_api_info_json(api_info)


def pretest_hook(name, phase):
    def pretest_info_dump_hook(module, in_feat, out_feat):
        pretest_info_dump(name, out_feat, module, phase)
        if hasattr(module, "input_args"):
            del module.input_args
        if hasattr(module, "input_kwargs"):
            del module.input_kwargs
    return pretest_info_dump_hook
