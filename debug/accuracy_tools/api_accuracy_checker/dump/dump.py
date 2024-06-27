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
import os
import time
import torch.distributed as dist

from api_accuracy_checker.dump.api_info import ForwardAPIInfo, BackwardAPIInfo
from api_accuracy_checker.dump.info_dump import write_api_info_json, initialize_output_json
from api_accuracy_checker.common.utils import print_error_log, CompareException, print_info_log, \
    get_tensor_rank, logger, Const, WarningManager
from api_accuracy_checker.hook_module.register_hook import initialize_hook
from api_accuracy_checker.common.config import msCheckerConfig

if msCheckerConfig.is_online:
    from api_accuracy_checker.tensor_transport_layer.attl import ATTL, ATTLConfig, ApiData


def set_dump_switch(switch):
    if switch not in ["ON", "OFF"]:
        print_error_log("Please set switch with 'ON' or 'OFF'.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    if switch == "ON":
        initialize_hook(pretest_hook)
        initialize_output_json()
    DumpUtil.set_dump_switch(switch)


def check_dataloader_status():
    if msCheckerConfig.enable_dataloader:
        error_info = ("If you want to use this  function, set enable_dataloader "
                      "in the accuracy_tools/api_accuracy_check/config.yaml "
                      "to False first")
        raise CompareException(CompareException.INVALID_PARAM_ERROR, error_info)


def start():
    wm = WarningManager()
    wm.warn(message=Const.VERSION_MESSAGE, enable_warnings=True)
    check_dataloader_status()
    if not DumpUtil.get_dump_switch():
        DumpUtil.incr_iter_num_maybe_exit()


def stop():
    check_dataloader_status()
    DumpUtil.set_dump_switch("OFF")


def step():
    check_dataloader_status()
    DumpUtil.call_num += 1


class DumpUtil(object):
    dump_switch = None
    call_num = 0
    phase = "all"
    rank_list = msCheckerConfig.rank_list
    attl = None
    if msCheckerConfig.is_online and not msCheckerConfig.is_benchmark_device:
        attl_config = ATTLConfig(False, connect_ip=msCheckerConfig.host,
                                 connect_port=msCheckerConfig.port,
                                 nfs_path=msCheckerConfig.nfs_path if msCheckerConfig.nfs_path else None)
        need_dump = dist.get_rank() in msCheckerConfig.rank_list if dist.is_initialized() else True
        attl = ATTL('npu', attl_config, need_dump=need_dump)
        if msCheckerConfig.nfs_path:
            attl.upload("start")

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
            if msCheckerConfig.is_online:
                if msCheckerConfig.nfs_path:
                    DumpUtil.attl.upload("end")
                elif DumpUtil.attl.socket_manager is not None:
                    logger.debug(f"进程{os.getpid()} 已完成,准备发送STOP信号")
                    DumpUtil.attl.socket_manager.send_stop_signal()
                else:
                    while True:
                        time.sleep(2)
            raise Exception("Model pretest: exit after iteration {}".format(DumpUtil.call_num - 1))
        else:
            set_dump_switch("OFF")


def pretest_info_dump(name, out_feat, module, phase):
    if not DumpUtil.get_dump_switch():
        return
    name = name.replace('*', Const.DELIMITER)
    if phase == Const.FORWARD:
        api_info = ForwardAPIInfo(name, module.input_args, module.input_kwargs)
    elif phase == Const.BACKWARD:
        api_info = BackwardAPIInfo(name, out_feat)
    else:
        msg = "Unexpected training phase {}.".format(phase)
        print_error_log(msg)
        raise NotImplementedError(msg)
    print_info_log(f"tools is dumping api: {name}" + " " * 10, end='\r')
    write_api_info_json(api_info)


def pretest_real_data_transport(name, out_feat, module, phase):
    if not DumpUtil.get_dump_switch():
        return
    name = name.replace('*', Const.DELIMITER)
    if phase == Const.FORWARD and (DumpUtil.phase == "all" or DumpUtil.phase == phase):
        cur_rank = get_tensor_rank(module.input_args, out_feat)
        if cur_rank not in DumpUtil.rank_list:
            return
        api_data = ApiData(name, module.input_args, module.input_kwargs, out_feat, DumpUtil.call_num, cur_rank)
        print_info_log(f"tools is dumping api: {api_data.name}, rank: {cur_rank}")
        if "device" in api_data.kwargs:
            api_data.kwargs.pop("device")
        if msCheckerConfig.nfs_path:
            DumpUtil.attl.upload(api_data)
        else:
            DumpUtil.attl.send(api_data)


def pretest_hook(name, phase):
    def pretest_info_dump_hook(module, in_feat, out_feat):
        if msCheckerConfig.is_online:
            pretest_real_data_transport(name, out_feat, module, phase)
        else:
            pretest_info_dump(name, out_feat, module, phase)
        if hasattr(module, "input_args"):
            del module.input_args
        if hasattr(module, "input_kwargs"):
            del module.input_kwargs

    return pretest_info_dump_hook
