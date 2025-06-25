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
import re
import sys
import time
import gc
from collections import namedtuple

try:
    import torch_npu
except ImportError:
    is_gpu = True
    current_device = "cuda"
else:
    is_gpu = False
    current_device = "npu"

import torch
from tqdm import tqdm

from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import BackwardMessage, UtDataInfo, \
    get_validated_result_csv_path, get_validated_details_csv_path, exec_api, record_skip_info, is_unsupported_api
from msprobe.pytorch.api_accuracy_checker.run_ut.data_generate import gen_api_params, gen_args
from msprobe.pytorch.api_accuracy_checker.common.utils import api_info_preprocess, \
    initialize_save_path, UtDataProcessor, extract_basic_api_segments, ApiData
from msprobe.pytorch.api_accuracy_checker.compare.compare import Comparator
from msprobe.pytorch.api_accuracy_checker.compare.compare_column import CompareColumn
from msprobe.pytorch.api_accuracy_checker.common.config import CheckerConfig
from msprobe.pytorch.common.parse_json import parse_json_info_forward_backward
from msprobe.core.common.file_utils import FileChecker, change_mode, \
    create_directory, get_json_contents, read_csv, check_file_or_directory_path
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.pt_config import parse_json_config
from msprobe.core.common.const import Const, FileCheckConst, CompareConst
from msprobe.core.common.utils import safe_get_value, CompareException, is_int, check_op_str_pattern_valid
from msprobe.pytorch.common.utils import seed_all
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.attl import ATTL, ATTLConfig, move2device_exec
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch import ConsumerDispatcher
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import generate_cpu_params, generate_device_params, \
    ExecParams


current_time = time.strftime("%Y%m%d%H%M%S")
UT_ERROR_DATA_DIR = 'ut_error_data' + current_time
RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + ".csv"


not_backward_list = ['repeat_interleave']
unsupported_backward_list = ['masked_select']
unsupported_api_list = ["to", "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", 
                        "empty_with_format"]


tqdm_params = {
    'smoothing': 0,  # 平滑进度条的预计剩余时间，取值范围0到1
    'desc': 'Processing',  # 进度条前的描述文字
    'leave': True,  # 迭代完成后保留进度条的显示
    'ncols': 75,  # 进度条的固定宽度
    'mininterval': 0.1,  # 更新进度条的最小间隔秒数
    'maxinterval': 1.0,  # 更新进度条的最大间隔秒数
    'miniters': 1,  # 更新进度条之间的最小迭代次数
    'ascii': None,  # 根据环境自动使用ASCII或Unicode字符
    'unit': 'it',  # 迭代单位
    'unit_scale': True,  # 自动根据单位缩放
    'dynamic_ncols': True,  # 动态调整进度条宽度以适应控制台
    'bar_format': '{l_bar}{bar}| {n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'  # 自定义进度条输出格式
}


seed_all()


def run_ut(config):
    logger.info("start UT test")
    if config.online_config.is_online:
        logger.info(f"UT task result will be saved in {config.result_csv_path}".replace(".csv", "_rank*.csv"))
        logger.info(f"UT task details will be saved in {config.details_csv_path}".replace(".csv", "_rank*.csv"))
    else:
        logger.info(f"UT task result will be saved in {config.result_csv_path}")
        logger.info(f"UT task details will be saved in {config.details_csv_path}")

    if config.save_error_data:
        logger.info(f"UT task error_data will be saved in {config.error_data_path}")
    compare = Comparator(config.result_csv_path, config.details_csv_path, config.is_continue_run_ut, config=config)

    if config.online_config.is_online:
        run_api_online(config, compare)
    else:
        csv_df = read_csv(config.result_csv_path)
        try:
            api_name_set = {row[0] for row in csv_df.itertuples(index=False, name=None)}
        except IndexError:
            logger.error(f"Read {config.result_csv_path} error, api_name_set is empty.")
            api_name_set = set()
        run_api_offline(config, compare, api_name_set)
    for result_csv_path, details_csv_path in zip(compare.save_path_list, compare.detail_save_path_list):
        change_mode(result_csv_path, FileCheckConst.DATA_FILE_AUTHORITY)
        change_mode(details_csv_path, FileCheckConst.DATA_FILE_AUTHORITY)
        logger.info(f"UT task result csv is saved in {result_csv_path}")
        logger.info(f"UT task details csv is saved in {details_csv_path}")
    compare.print_pretest_result()


def run_api_offline(config, compare, api_name_set):
    err_column = CompareColumn()
    for _, (api_full_name, api_info_dict) in enumerate(tqdm(config.forward_content.items(), **tqdm_params)):
        check_op_str_pattern_valid(api_full_name)
        if api_full_name in api_name_set:
            continue
        if is_unsupported_api(api_full_name):
            skip_message = f"API {api_full_name} not support for run ut. SKIP."
            compare_alg_results = err_column.to_column_value(CompareConst.SKIP, skip_message)
            record_skip_info(api_full_name, compare, compare_alg_results)
            continue
        _, api_name = extract_basic_api_segments(api_full_name)
        if not api_name:
            err_message = f"API {api_full_name} not support for run ut. SKIP."
            logger.error(err_message)
            compare_alg_results = err_column.to_column_value(CompareConst.SKIP, err_message)
            record_skip_info(api_full_name, compare, compare_alg_results)
            continue
        try:
            if blacklist_and_whitelist_filter(api_name, config.black_list, config.white_list):
                skip_message = f"API {api_name} in black list or not in white list. SKIP."
                logger.info(skip_message)
                compare_alg_results = err_column.to_column_value(CompareConst.SKIP, skip_message)
                record_skip_info(api_full_name, compare, compare_alg_results)
                continue
            data_info = run_torch_api(api_full_name, config.real_data_path, config.backward_content, api_info_dict)
            is_fwd_success, is_bwd_success = compare.compare_output(api_full_name, data_info)
            if config.save_error_data:
                do_save_error_data(api_full_name, data_info, config.error_data_path, is_fwd_success, is_bwd_success)
        except Exception as err:
            if "expected scalar type Long" in str(err):
                logger.warning(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               "'int32_to_int64' list in accuracy_tools/msprobe/core/common/const.py file.")
            else:
                logger.error(f"Run {api_full_name} UT Error: %s" % str(err))
            compare_alg_results = err_column.to_column_value(CompareConst.SKIP, str(err))
            record_skip_info(api_full_name, compare, compare_alg_results)
        finally:
            if is_gpu:
                torch.cuda.empty_cache()
            else:
                torch.npu.empty_cache()
            gc.collect()


def run_api_online(config, compare):
    attl = init_attl(config.online_config)
    dispatcher = ConsumerDispatcher(compare=compare)
    dispatcher.start(handle_func=run_torch_api_online, config=config)

    def tcp_communication_flow():
        while True:
            api_data = attl.recv()
            if api_data == 'STOP_':
                continue
            if api_data == 'KILL_':
                time.sleep(1)
                logger.info("==========接收到STOP信号==========")
                dispatcher.stop()
                attl.stop_serve()
                time.sleep(1)
                break
            if not isinstance(api_data, ApiData):
                continue
            api_full_name = api_data.name
            _, api_name = extract_basic_api_segments(api_full_name)
            if blacklist_and_whitelist_filter(api_name, config.black_list, config.white_list):
                continue
            if api_data.rank in config.online_config.rank_list:
                dispatcher.update_consume_queue(api_data)

    def shared_storage_communication_flow():
        flag_num = -1
        while True:
            api_data = attl.download()
            if api_data == "start":
                if flag_num == -1:
                    flag_num += 1
                flag_num += 1
            if api_data == "end":
                flag_num -= 1
            if flag_num == 0:
                dispatcher.stop()
                break
            if not isinstance(api_data, ApiData):
                continue
            api_full_name = api_data.name
            _, api_name = extract_basic_api_segments(api_full_name)
            if blacklist_and_whitelist_filter(api_name, config.black_list, config.white_list):
                continue
            if api_data.rank in config.online_config.rank_list:
                dispatcher.update_consume_queue(api_data)

    if config.online_config.nfs_path:
        shared_storage_communication_flow()
    else:
        tcp_communication_flow()


def blacklist_and_whitelist_filter(api_name, black_list, white_list):
    """
    run api(api_name) if api_name not in black_list and in white_list.
    If api is both in black_list and black_list, black_list first.
    return: False for exec api, True for not exec
    """
    black_list.extend(unsupported_api_list)
    if black_list and api_name in black_list:
        return True
    if white_list and api_name not in white_list:
        return True
    return False


def do_save_error_data(api_full_name, data_info, error_data_path, is_fwd_success, is_bwd_success):
    if not is_fwd_success or not is_bwd_success:
        processor = UtDataProcessor(error_data_path)
        for element in data_info.in_fwd_data_list:
            processor.save_tensors_in_element(api_full_name + '.forward.input', element)
        processor.save_tensors_in_element(api_full_name + '.forward.output.bench', data_info.bench_output)
        processor.save_tensors_in_element(api_full_name + '.forward.output.device', data_info.device_output)
        processor.save_tensors_in_element(api_full_name + '.backward.input', data_info.grad_in)
        processor.save_tensors_in_element(api_full_name + '.backward.output.bench', data_info.bench_grad)
        processor.save_tensors_in_element(api_full_name + '.backward.output.device', data_info.device_grad)


def run_torch_api(api_full_name, real_data_path, backward_content, api_info_dict):
    in_fwd_data_list = []
    backward_message = ''
    api_type, api_name = extract_basic_api_segments(api_full_name)
    args, kwargs, output_dtype = get_api_info(api_info_dict, api_name, real_data_path)
    need_grad = check_need_grad(api_info_dict)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    need_backward = api_full_name in backward_content
    if not need_grad:
        logger.warning("%s %s" % (api_full_name, BackwardMessage.UNSUPPORT_BACKWARD_MESSAGE))
        backward_message += BackwardMessage.UNSUPPORT_BACKWARD_MESSAGE
    if api_name in not_backward_list:
        need_grad = False
        logger.info("%s %s" % (api_full_name, BackwardMessage.NO_BACKWARD_RESULT_MESSAGE))
        backward_message += BackwardMessage.NO_BACKWARD_RESULT_MESSAGE
    if api_name in unsupported_backward_list:
        need_grad = False
        logger.info("%s %s" % (api_full_name, BackwardMessage.UNSUPPORT_API_MESSAGE))
        backward_message += BackwardMessage.UNSUPPORT_API_MESSAGE
    need_backward = need_backward and need_grad
    
    device_info_kwargs = kwargs.get(Const.DEVICE)
    if device_info_kwargs and device_info_kwargs.get(Const.VALUE):
        kwargs[Const.DEVICE] = current_device
    device_args, device_kwargs = generate_device_params(args, kwargs, need_backward, api_name)
    if kwargs.get(Const.DEVICE):
        del kwargs[Const.DEVICE]
    cpu_params = generate_cpu_params(args, kwargs, need_backward, api_name)
    cpu_args, cpu_kwargs = cpu_params.cpu_args, cpu_params.cpu_kwargs
    autocast_dtype, is_autocast = cpu_params.autocast_dtype, cpu_params.is_autocast
    if not is_autocast and output_dtype:
        is_autocast = autocast_dtype != output_dtype
        autocast_dtype = output_dtype
    bench_grad_out, device_grad_out = None, None
    cpu_exec_params = ExecParams(api_type, api_name, Const.CPU_LOWERCASE, cpu_args, cpu_kwargs, False, autocast_dtype)
    out = exec_api(cpu_exec_params)
    device_exec_params = ExecParams(api_type, api_name, current_device, device_args, device_kwargs, is_autocast,
                                     autocast_dtype)
    device_out = exec_api(device_exec_params)
    current_path = os.path.dirname(os.path.realpath(__file__))
    ut_setting_path = os.path.join(current_path, "torch_ut_setting.json")
    api_setting_dict = get_json_contents(ut_setting_path)
    grad_input_index = api_setting_dict.get(api_name)
    grad_index = None
    grad, bench_grad = None, None
    if grad_input_index is not None:
        grad_index = grad_input_index.get('grad_index')

    if need_backward and out is not None:
        if need_to_backward(grad_index, out):
            backward_args = backward_content[api_full_name].get("input")
            func_options = {
                'real_data_path': real_data_path
            }
            grad = gen_args(backward_args, api_name, func_options)
            grad = safe_get_value(grad, 0, "grad")
            grad_params = generate_cpu_params(grad, {}, False, api_name)
            bench_grad = grad_params.cpu_args
            bench_grad_out = run_backward(cpu_args, bench_grad, grad_index, out)
            device_grad = grad.clone().detach().to(current_device)
            device_grad_out = run_backward(device_args, device_grad, grad_index, device_out)
        else:
            backward_message += BackwardMessage.MULTIPLE_BACKWARD_MESSAGE
    if api_name == "npu_fusion_attention":
        out = safe_get_value(out, 0, "out")
        device_out = safe_get_value(device_out, 0, "device_out")

    return UtDataInfo(bench_grad_out, device_grad_out, device_out, out, bench_grad, in_fwd_data_list, backward_message)


def run_torch_api_online(api_full_name, api_data, backward_content):
    in_fwd_data_list = []
    api_type, api_name = extract_basic_api_segments(api_full_name)
    args, kwargs, out = api_data.args, api_data.kwargs, api_data.result
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    if kwargs.get("device"):
        del kwargs["device"]

    device_exec_params = ExecParams(api_type, api_name, current_device, args, kwargs, False, None)
    device_out = exec_api(device_exec_params)
    device_out = move2device_exec(device_out, "cpu")
    return UtDataInfo(None, None, out, device_out, None, in_fwd_data_list, None, rank=api_data.rank)


def check_need_grad(api_info_dict):
    need_grad = True
    if api_info_dict.get(Const.INPUT_KWARGS) and "out" in api_info_dict.get(Const.INPUT_KWARGS):
        need_grad = False
    return need_grad


def get_api_info(api_info_dict, api_name, real_data_path):
    convert_type, api_info_dict = api_info_preprocess(api_name, api_info_dict)
    need_grad = check_need_grad(api_info_dict)
    args, kwargs, output_dtype = gen_api_params(api_info_dict, api_name, need_grad, convert_type, real_data_path)
    return args, kwargs, output_dtype


def need_to_backward(grad_index, out):
    if grad_index is None and isinstance(out, (list, tuple)):
        return False
    return True


def run_backward(args, grad, grad_index, out):
    if grad_index is not None:
        if not is_int(grad_index):
            logger.error(f"{grad_index} dtype is not int")
            raise TypeError(f"{grad_index} dtype is not int")
        if grad_index >= len(out):
            logger.error(f"Run backward error when grad_index is {grad_index}")
            raise IndexError(f"Run backward error when grad_index is {grad_index}")
        out[grad_index].backward(grad)
    else:
        out.backward(grad)

    grad_out = extract_tensors_grad(args)

    return grad_out


def extract_tensors_grad(args, depth=0):
    if depth > Const.MAX_DEPTH:
        logger.error("The depth of arg_in is too large, please check the arg_in.")
        raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
    grads = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            grads.append(arg.grad)
        elif isinstance(arg, (list, tuple)):
            grads.extend(extract_tensors_grad(arg, depth+1))
    return grads


def initialize_save_error_data(error_data_path):
    create_directory(error_data_path)
    error_data_path_checker = FileChecker(error_data_path, FileCheckConst.DIR,
                                          ability=FileCheckConst.WRITE_ABLE)
    error_data_path = error_data_path_checker.common_check()
    error_data_path = initialize_save_path(error_data_path, UT_ERROR_DATA_DIR)
    return error_data_path


def init_attl(config):
    """config: OnlineConfig"""
    attl = ATTL('gpu', ATTLConfig(is_benchmark_device=True,
                                  connect_ip=config.host,
                                  connect_port=config.port,
                                  nfs_path=config.nfs_path,
                                  tls_path=config.tls_path))
    return attl


def _run_ut_parser(parser):
    parser.add_argument("-api_info", "--api_info_file", dest="api_info_file", default="", type=str,
                        help="<Optional> The api param tool result file: generate from api param tool, "
                             "a json file.",
                        required=False)
    parser.add_argument("-o", "--out_path", dest="out_path", default="", type=str,
                        help="<optional> The ut task result out path.",
                        required=False)
    parser.add_argument('-save_error_data', dest="save_error_data", action="store_true",
                        help="<optional> Save compare failed api output.", required=False)
    parser.add_argument("-j", "--jit_compile", dest="jit_compile", action="store_true",
                        help="<optional> whether to turn on jit compile", required=False)

    class UniqueDeviceAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            unique_values = set(values)
            if len(values) != len(unique_values):
                parser.error("device id must be unique")
            for device_id in values:
                if not 0 <= device_id:
                    parser.error("device id must be greater than or equal to 0")
            setattr(namespace, self.dest, values)

    parser.add_argument("-d", "--device", dest="device_id", nargs='+', type=int,
                        help="<optional> set device id to run ut, must be unique and in range 0-7",
                        default=[0], required=False, action=UniqueDeviceAction)
    parser.add_argument("-csv_path", "--result_csv_path", dest="result_csv_path", default="", type=str,
                        help="<optional> The path of accuracy_checking_result_{timestamp}.csv, "
                             "when run ut is interrupted, enter the file path to continue run ut.",
                        required=False)
    parser.add_argument("-f", "--filter_api", dest="filter_api", action="store_true",
                        help="<optional> Whether to filter the api in the api_info_file.", required=False)
    parser.add_argument("-config", "--config_path", dest="config_path", default="", type=str,
                        help="<optional> The path of config.json", required=False)


def preprocess_forward_content(forward_content):
    processed_content = {}
    base_keys_variants = {}
    arg_cache = {}

    for key, value in forward_content.items():
        check_op_str_pattern_valid(key)
        base_key = key.rsplit(Const.SEP, 1)[0]

        if key not in arg_cache:
            filtered_new_args = [
                {k: v for k, v in arg.items() if k not in ['Max', 'Min']}
                for arg in value['input_args']
                if isinstance(arg, dict)
            ]
            arg_cache[key] = (filtered_new_args, value['input_kwargs'])

        filtered_new_args, new_kwargs = arg_cache[key]

        if base_key not in base_keys_variants:
            processed_content[key] = value
            base_keys_variants[base_key] = {key}
        else:
            is_duplicate = False
            for variant in base_keys_variants.get(base_key, []):
                try:
                    existing_args, existing_kwargs = arg_cache.get(variant)
                except KeyError as e:
                    logger.error(f"KeyError: {e} when processing {key}")
                if existing_args == filtered_new_args and existing_kwargs == new_kwargs:
                    is_duplicate = True
                    break

            if not is_duplicate:
                processed_content[key] = value
                base_keys_variants[base_key].add(key)

    return processed_content


def _run_ut(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    run_ut_command(args)
    

def checked_online_config(online_config):
    if not online_config.is_online:
        return
    if not isinstance(online_config.is_online, bool):
        raise ValueError("is_online must be bool type")
    # rank_list
    if not isinstance(online_config.rank_list, list):
        raise ValueError("rank_list must be a list")
    if online_config.rank_list and not all(isinstance(rank, int) for rank in online_config.rank_list):
        raise ValueError("All elements in rank_list must be integers")

    # nfs_path
    if online_config.nfs_path:
        check_file_or_directory_path(online_config.nfs_path, isdir=True)
        return
    # tls_path
    if online_config.tls_path:
        check_file_or_directory_path(online_config.tls_path, isdir=True)
        check_file_or_directory_path(os.path.join(online_config.tls_path, "server.key"))
        check_file_or_directory_path(os.path.join(online_config.tls_path, "server.crt"))
        check_file_or_directory_path(os.path.join(online_config.tls_path, "ca.crt"))
        crl_path = os.path.join(online_config.tls_path, "crl.pem")
        if os.path.exists(crl_path):
            check_file_or_directory_path(crl_path)

    # host and port
    if not isinstance(online_config.host, str) or not re.match(Const.ipv4_pattern, online_config.host):
        raise Exception(f"host: {online_config.host} is invalid.")
    if not isinstance(online_config.port, int) or not (0 < online_config.port <= 65535):
        raise Exception(f"port: {online_config.port} is invalid, port range 0-65535.")


def run_ut_command(args):
    if args.config_path:
        config_path_checker = FileChecker(args.config_path, FileCheckConst.FILE, 
                                          FileCheckConst.READ_ABLE, FileCheckConst.JSON_SUFFIX)
        checked_config_path = config_path_checker.common_check()
        _, task_config = parse_json_config(checked_config_path, Const.RUN_UT)
        checker_config = CheckerConfig(task_config)
    else:
        checker_config = CheckerConfig()
    
    if not checker_config.is_online and not args.api_info_file:
        logger.error("Please provide api_info_file for offline run ut.")
        raise Exception("Please provide api_info_file for offline run ut.")

    if not is_gpu:
        torch.npu.set_compile_mode(jit_compile=args.jit_compile)
        if args.jit_compile:
            torch.npu.config.allow_internal_format = True
        else:
            torch.npu.config.allow_internal_format = False
    used_device = current_device + ":" + str(args.device_id[0])
    try:
        if is_gpu:
            torch.cuda.set_device(used_device)
        else:
            torch.npu.set_device(used_device)
    except Exception as error:
        logger.error(f"Set device id failed. device id is: {args.device_id}")
        raise NotImplementedError from error

    # 在线预检场景下，不需要外出输出api信息，forward_content, backward_content, real_data_path设置为None
    # 离线场景下，forward_content, backward_content, real_data_path从api_info_file中解析
    forward_content, backward_content, real_data_path = None, None, None
    if args.api_info_file:
        api_info_file_checker = FileChecker(file_path=args.api_info_file, path_type=FileCheckConst.FILE, 
                                            ability=FileCheckConst.READ_ABLE, file_type=FileCheckConst.JSON_SUFFIX)
        checked_api_info = api_info_file_checker.common_check()
        forward_content, backward_content, real_data_path = parse_json_info_forward_backward(checked_api_info)
        if real_data_path:
            dump_path = os.path.dirname(checked_api_info)
            real_data_path = os.path.join(dump_path, Const.DUMP_TENSOR_DATA)
        if args.filter_api:
            logger.info("Start filtering the api in the api_info_file.")
            forward_content = preprocess_forward_content(forward_content)
            logger.info("Finish filtering the api in the api_info_file.")

    out_path = args.out_path if args.out_path else Const.DEFAULT_PATH
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    save_error_data = args.save_error_data

    result_csv_path = os.path.join(out_path, RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, DETAILS_FILE_NAME)
    if args.result_csv_path:
        result_csv_path = get_validated_result_csv_path(args.result_csv_path, 'result')
        details_csv_path = get_validated_details_csv_path(result_csv_path)

    error_data_path = checker_config.error_data_path
    if save_error_data:
        if args.result_csv_path:
            parts_by_dot = result_csv_path.split(Const.SEP)
            if len(parts_by_dot) < 2 or not parts_by_dot[0]:
                raise ValueError("result_csv_path does not contain a valid file name with an extension.")
            file_name_part = parts_by_dot[0]
            parts_by_underscore = file_name_part.split(Const.REPLACEMENT_CHARACTER)
            if len(parts_by_underscore) < 2:
                raise ValueError("File name part does not contain enough '_' separated segments.")
            time_info = parts_by_underscore[-1]

            global UT_ERROR_DATA_DIR
            UT_ERROR_DATA_DIR = 'ut_error_data' + time_info
        error_data_path = initialize_save_error_data(error_data_path)
    online_config = checker_config.get_online_config()
    checked_online_config(online_config)
    config_params = {
        'forward_content': forward_content,
        'backward_content': backward_content,
        'result_csv_path': result_csv_path,
        'details_csv_path': details_csv_path,
        'save_error_data': save_error_data,
        'is_continue_run_ut': args.result_csv_path,
        'real_data_path': real_data_path,
        'error_data_path': error_data_path
    }
    run_ut_config = checker_config.get_run_ut_config(**config_params)
    run_ut(run_ut_config)
    logger.info("UT task completed.")


if __name__ == '__main__':
    _run_ut()
