#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import argparse
import os
import sys

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False
import torch
from tqdm import tqdm
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut import generate_device_params, get_api_info
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import exec_api
from msprobe.core.common.file_utils import check_link
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.parse_json import parse_json_info_forward_backward
from msprobe.core.common.const import Const


def check_tensor_overflow(x):
    if isinstance(x, torch.Tensor) and x.numel() != 0 and x.dtype != torch.bool:
        if len(x.shape) == 0:
            tensor_max = x.cpu().detach().float().numpy().tolist()
            tensor_min = tensor_max
        else:
            tensor_max = torch.max(x).cpu().detach().float().numpy().tolist()
            tensor_min = torch.min(x).cpu().detach().float().numpy().tolist()
        # inf
        if tensor_max == float('inf') or tensor_min == float('-inf'):
            return True
        # nan
        elif tensor_max != tensor_max or tensor_min != tensor_min:
            return True
        else:
            return False
    elif isinstance(x, bool) or isinstance(x, int) or isinstance(x, float):
        if x == float('inf') or x == float('-inf') or x != x:
            return True
        else:
            return False
    else:
        return False


def check_data_overflow(x):
    if isinstance(x, (tuple, list)) and x:
        for _, item in enumerate(x):
            if check_data_overflow(item):
                return True
        return False
    else:
        return check_tensor_overflow(x)


def run_overflow_check(forward_file):
    logger.info("start UT test")
    forward_content, _, real_data_path = parse_json_info_forward_backward(forward_file)
    for api_full_name, api_info_dict in tqdm(forward_content.items()):
        try:
            run_torch_api(api_full_name, api_info_dict, real_data_path)
        except Exception as err:
            _, api_name, _ = api_full_name.split(Const.SEP)
            if "not implemented for 'Half'" in str(err):
                logger.warning(f"API {api_name} not support half tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'fp16_to_fp32' list in accuracy_tools/api_accuracy_check/common/utils.py file.")
            elif "expected scalar type Long" in str(err):
                logger.warning(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file.")
            else:
                logger.error(f"Run {api_full_name} UT Error: %s" % str(err))


def run_torch_api(api_full_name, api_info_dict, real_data_path):
    torch.npu.clear_npu_overflow_flag()
    api_type, api_name, _ = api_full_name.split(Const.SEP)
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name, real_data_path)
    if not need_grad:
        logger.warning("%s function with out=... arguments don't support automatic differentiation, skip backward." 
                       % api_full_name)
    npu_args, npu_kwargs = generate_device_params(args, kwargs, False, api_name)
    if kwargs.get("device"):
        del kwargs["device"]
    out = exec_api(api_type, api_name, Const.CPU_LOWERCASE, args, kwargs)
    npu_out = exec_api(api_type, api_name, Const.NPU_LOWERCASE, npu_args, npu_kwargs)
    if out is None and npu_out is None:
        logger.warning("The %s overflow is a normal overflow, out and npu_out is None." % api_full_name)
        return

    cpu_overflow = check_data_overflow(out)
    npu_overflow = torch_npu.npu.utils.npu_check_overflow(npu_out)
    if cpu_overflow == npu_overflow:
        logger.warning("The %s overflow is a normal overflow." % api_full_name)
    else:
        logger.warning("The %s overflow is an abnormal overflow." % api_full_name)
    return


def _run_overflow_check_parser(parser):
    parser.add_argument("-api_info", "--api_info_file", dest="api_info_file", default="",
                        help="<Required> The api param tool result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-j", "--jit_compile", dest="jit_compile", help="<optional> whether to turn on jit compile",
                        default=False, required=False)
    parser.add_argument("-d", "--device", dest="device_id", type=int, help="<optional> set NPU device id to run ut",
                        default=0, required=False)


def _run_overflow_check(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _run_overflow_check_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    _run_overflow_check_command(args)


def _run_overflow_check_command(args):
    torch.npu.set_compile_mode(jit_compile=args.jit_compile)
    npu_device = "npu:" + str(args.device_id)
    check_link(args.api_info_file)
    api_info = os.path.realpath(args.api_info_file)
    try:
        torch.npu.set_device(npu_device)
    except Exception as error:
        logger.error(f"Set NPU device id failed. device id is: {args.device_id}")
        raise NotImplementedError from error
    run_overflow_check(api_info)


if __name__ == '__main__':
    _run_overflow_check()
    logger.info("UT task completed.")
