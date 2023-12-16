import argparse
import os
import csv
import re
import sys
import time
from collections import namedtuple
try:
    import torch_npu
except ImportError:
    is_gpu = True
    current_device = "cuda"
else:
    is_gpu = False
    current_device = "npu"
import yaml
import torch
from tqdm import tqdm
from api_accuracy_checker.run_ut.data_generate import gen_api_params, gen_args
from api_accuracy_checker.common.utils import print_info_log, print_warn_log, get_json_contents, api_info_preprocess, \
    print_error_log, check_file_or_directory_path, initialize_save_path, Const
from api_accuracy_checker.compare.compare import Comparator
from api_accuracy_checker.hook_module.wrap_tensor import TensorOPTemplate
from api_accuracy_checker.hook_module.wrap_functional import FunctionalOPTemplate
from api_accuracy_checker.hook_module.wrap_torch import TorchOPTemplate
from api_accuracy_checker.run_ut.ut_api_info import UtAPIInfo
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.compare.compare_utils import CompareConst

from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen, FileCheckConst, FileChecker, \
    change_mode, check_file_suffix, check_link

current_time = time.strftime("%Y%m%d%H%M%S")
UT_ERROR_DATA_DIR = 'ut_error_data' + current_time
RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + ".csv"
RunUTConfig = namedtuple('RunUTConfig', ['forward_content', 'backward_content', 'result_csv_path', 'details_csv_path',
                                         'save_error_data', 'is_continue_run_ut', 'test_result_cnt'])
not_backward_list = ['repeat_interleave']

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


def exec_api(api_type, api_name, args, kwargs):
    if api_type == "Functional":
        functional_api = FunctionalOPTemplate(api_name, str, False)
        out = functional_api.forward(*args, **kwargs)
    if api_type == "Tensor":
        tensor_api = TensorOPTemplate(api_name, str, False)
        out = tensor_api.forward(*args, **kwargs)
    if api_type == "Torch":
        torch_api = TorchOPTemplate(api_name, str, False)
        out = torch_api.forward(*args, **kwargs)
    return out


def deal_detach(arg, to_detach=True):
    return arg.detach() if to_detach else arg


def deal_dtype(arg, raise_dtype=None):
    if raise_dtype is None or arg.dtype not in Const.RAISE_PRECISION or raise_dtype == arg.dtype:
        return arg
    return arg.type(raise_dtype)


def generate_device_params(input_args, input_kwargs, need_backward):
    def recursive_arg_to_device(arg_in, to_detach=True):
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_device(arg, to_detach) for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if need_backward and arg_in.requires_grad:
                arg_in = deal_detach(arg_in.clone(), to_detach).to(current_device).requires_grad_()
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.type_as(arg_in)
                arg_in.retain_grad()
                return arg_in
            else:
                return deal_detach(arg_in.clone(), to_detach).to(current_device)
        else:
            return arg_in

    device_args = recursive_arg_to_device(input_args)
    device_kwargs = {key: recursive_arg_to_device(value, key != "out") for key, value in input_kwargs.items()}
    return device_args, device_kwargs


def generate_cpu_params(input_args, input_kwargs, need_backward):
    def recursive_arg_to_cpu(arg_in, to_detach=True, raise_dtype=None):
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_cpu(arg, to_detach, raise_dtype=raise_dtype) for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if need_backward and arg_in.requires_grad:
                arg_in = deal_detach(deal_dtype(arg_in.clone(), raise_dtype), to_detach).requires_grad_()
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.type_as(arg_in)
                arg_in.retain_grad()
                return arg_in
            else:
                return deal_detach(deal_dtype(arg_in.clone(), raise_dtype=raise_dtype), to_detach)
        else:
            return arg_in

    def recursive_find_dtypes(arg_in):
        if isinstance(arg_in, (list, tuple)):
            return set().union(*tuple(recursive_find_dtypes(arg) for arg in arg_in))
        elif isinstance(arg_in, torch.Tensor) and arg_in.dtype in Const.RAISE_PRECISION:
            return set([arg_in.dtype])
        return set()

    raise_dtype = None
    need_raise_dtypes = recursive_find_dtypes(input_args)
    if len(need_raise_dtypes) == 1:
        raise_dtype = Const.RAISE_PRECISION.get(need_raise_dtypes.pop())
    elif len(need_raise_dtypes) >= 2:
        raise_dtype = torch.float32

    cpu_args = recursive_arg_to_cpu(input_args, raise_dtype=raise_dtype)
    cpu_kwargs = {key: recursive_arg_to_cpu(value, key != "out") for key, value in input_kwargs.items()}
    return cpu_args, cpu_kwargs


def run_ut(config):
    print_info_log("start UT test")
    api_setting_dict = get_json_contents("torch_ut_setting.json")
    compare = Comparator(config.result_csv_path, config.details_csv_path, config.is_continue_run_ut,
                         config.test_result_cnt)
    with FileOpen(config.result_csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        api_name_set = {row[0] for row in csv_reader}
    for i, (api_full_name, api_info_dict) in enumerate(tqdm(config.forward_content.items(), **tqdm_params)):
        if api_full_name in api_name_set:
            continue
        try:
            if msCheckerConfig.white_list:
                [_, api_name, _] = api_full_name.split("*")
                if api_name not in set(msCheckerConfig.white_list):
                    continue
            data_info = run_torch_api(api_full_name, api_setting_dict, config.backward_content, api_info_dict)
            is_fwd_success, is_bwd_success = compare.compare_output(api_full_name,
                                                                    data_info.bench_out,
                                                                    data_info.device_out,
                                                                    data_info.bench_grad_out,
                                                                    data_info.device_grad_out)
            if config.save_error_data:
                do_save_error_data(api_full_name, data_info, is_fwd_success, is_bwd_success)
        except Exception as err:
            [_, api_name, _] = api_full_name.split("*")
            if "expected scalar type Long" in str(err):
                print_warn_log(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file.")
            else:
                print_error_log(f"Run {api_full_name} UT Error: %s" % str(err))
            compare.write_summary_csv((api_full_name, "SKIP", "SKIP", str(err)))
    change_mode(compare.save_path, FileCheckConst.DATA_FILE_AUTHORITY)
    change_mode(compare.detail_save_path, FileCheckConst.DATA_FILE_AUTHORITY)
    compare.print_pretest_result()


def do_save_error_data(api_full_name, data_info, is_fwd_success, is_bwd_success):
    if not is_fwd_success or not is_bwd_success:
        api_full_name = api_full_name.replace("*", ".")
        for element in data_info.in_fwd_data_list:
            UtAPIInfo(api_full_name + '.forward.input', element, UT_ERROR_DATA_DIR)
        UtAPIInfo(api_full_name + '.forward.output.bench', data_info.bench_out, UT_ERROR_DATA_DIR)
        UtAPIInfo(api_full_name + '.forward.output.device', data_info.device_out, UT_ERROR_DATA_DIR)
        UtAPIInfo(api_full_name + '.backward.input', data_info.grad_in, UT_ERROR_DATA_DIR)
        UtAPIInfo(api_full_name + '.backward.output.bench', data_info.bench_grad_out, UT_ERROR_DATA_DIR)
        UtAPIInfo(api_full_name + '.backward.output.device', data_info.device_grad_out, UT_ERROR_DATA_DIR)


def run_torch_api(api_full_name, api_setting_dict, backward_content, api_info_dict):
    in_fwd_data_list = []
    [api_type, api_name, _] = api_full_name.split("*")
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    need_backward = api_full_name in backward_content
    if not need_grad:
        print_warn_log("%s function with out=... arguments don't support automatic differentiation, skip backward."
                       % api_full_name)
    if api_name in not_backward_list:
        need_grad = False
        print_warn_log(
            "%s function backward result is None, skip backward." % api_full_name)
    need_backward = need_backward and need_grad
    if kwargs.get("device"):
        del kwargs["device"]
    cpu_args, cpu_kwargs = generate_cpu_params(args, kwargs, need_backward)
    device_args, device_kwargs = generate_device_params(args, kwargs, need_backward)
    grad_out, device_grad_out = None, None
    out = exec_api(api_type, api_name, cpu_args, cpu_kwargs)
    device_out = exec_api(api_type, api_name, device_args, device_kwargs)
    grad_input_index = api_setting_dict.get(api_name)
    grad_index = None
    grad = None
    if grad_input_index is not None:
        grad_index = grad_input_index.get('grad_index')

    if need_backward:
        grad_out, device_grad_out, grad, device_grad = run_backward(api_full_name, cpu_args, backward_content, grad_index, device_args,
                                                              device_out, out)
    if grad_index is not None:
        return UtDataInfo(grad_out, device_grad_out, device_out[grad_index], out[grad_index], grad, in_fwd_data_list)
    return UtDataInfo(grad_out, device_grad_out, device_out, out, grad, in_fwd_data_list)


def get_api_info(api_info_dict, api_name):
    convert_type, api_info_dict = api_info_preprocess(api_name, api_info_dict)
    need_grad = True
    if api_info_dict.get("kwargs") and "out" in api_info_dict.get("kwargs"):
        need_grad = False
    args, kwargs = gen_api_params(api_info_dict, need_grad, convert_type)
    return args, kwargs, need_grad


def run_backward(api_full_name, args, backward_content, grad_index, device_args, device_out, out):
    backward_args = backward_content[api_full_name]
    grad = gen_args(backward_args)[0]
    cpu_grad, _ = generate_cpu_params(grad, {}, False)
    if grad_index is not None:
        out[grad_index].backward(cpu_grad)
    elif isinstance(out, (list, tuple)):
        raise NotImplementedError("Multiple backward is not supported.")
    else:
        out.backward(cpu_grad)
    args_grad = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            args_grad.append(arg.grad)
    grad_out = args_grad
    device_grad = grad.clone().detach().to(current_device)
    if grad_index is not None:
        device_out[grad_index].backward(device_grad)
    else:
        device_out.backward(device_grad)
    device_args_grad = []
    for arg in device_args:
        if isinstance(arg, torch.Tensor):
            device_args_grad.append(arg.grad)
    device_grad_out = device_args_grad
    return grad_out, device_grad_out, grad, device_grad


def initialize_save_error_data():
    error_data_path_checker = FileChecker(msCheckerConfig.error_data_path, FileCheckConst.DIR,
                                          ability=FileCheckConst.WRITE_ABLE)
    error_data_path = error_data_path_checker.common_check()
    initialize_save_path(error_data_path, UT_ERROR_DATA_DIR)


def get_validated_result_csv_path(result_csv_path):
    result_csv_path_checker = FileChecker(result_csv_path, FileCheckConst.FILE, ability=FileCheckConst.READ_WRITE_ABLE,
                                          file_type=FileCheckConst.CSV_SUFFIX)
    validated_result_csv_path = result_csv_path_checker.common_check()
    result_csv_name = os.path.basename(validated_result_csv_path)
    pattern = r"^accuracy_checking_result_\d{14}\.csv$"
    if not re.match(pattern, result_csv_name):
        raise ValueError("When continue run ut, please do not modify the result csv name.")
    return validated_result_csv_path


def get_validated_details_csv_path(validated_result_csv_path):
    result_csv_name = os.path.basename(validated_result_csv_path)
    details_csv_name = result_csv_name.replace('result', 'details')
    details_csv_path = os.path.join(os.path.dirname(validated_result_csv_path), details_csv_name)
    details_csv_path_checker = FileChecker(details_csv_path, FileCheckConst.FILE,
                                           ability=FileCheckConst.READ_WRITE_ABLE, file_type=FileCheckConst.CSV_SUFFIX)
    validated_details_csv_path = details_csv_path_checker.common_check()
    return validated_details_csv_path


def get_statistics_from_result_csv(validated_result_csv_path):
    test_result_cnt = {
        "forward_fail_num": 0, "backward_fail_num": 0, "forward_and_backward_fail_num": 0, "success_num": 0,
        "total_num": 0, "forward_or_backward_fail_num": 0
    }
    with FileOpen(validated_result_csv_path, 'r') as file:
        reader = csv.reader(file)
        result_csv_rows = [row for row in reader]
    result_csv_name = os.path.basename(validated_result_csv_path)
    for item in result_csv_rows[1:]:
        if not isinstance(item, list) or len(item) < 3:
            raise ValueError("The number of columns in %s is incorrect" % result_csv_name)
        if item[1] not in ['True', 'False', CompareConst.NA, 'SKIP'] \
                or item[2] not in ['True', 'False', CompareConst.NA, 'SKIP']:
            raise ValueError("The value in the 2nd or 3rd column of %s is wrong, it must be TRUE, FALSE or N/A"
                             % result_csv_name)
        if item[1] == 'SKIP':
            continue
        test_result_cnt["total_num"] += 1
        if item[1] == 'True' and item[2] in ['True', 'N/A']:
            test_result_cnt['success_num'] += 1
        elif item[1] == 'False' and item[2] == 'False':
            test_result_cnt['forward_and_backward_fail_num'] += 1
        elif item[1] == 'False':
            test_result_cnt['forward_fail_num'] += 1
            test_result_cnt['forward_or_backward_fail_num'] += 1
        else:
            test_result_cnt['backward_fail_num'] += 1
            test_result_cnt['forward_or_backward_fail_num'] += 1
    return test_result_cnt


def _run_ut_parser(parser):
    parser.add_argument("-forward", "--forward_input_file", dest="forward_input_file", default="", type=str,
                        help="<Required> The api param tool forward result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-backward", "--backward_input_file", dest="backward_input_file", default="", type=str,
                        help="<Required> The api param tool backward result file: generate from api param tool, "
                             "a json file.",
                        required=False)
    parser.add_argument("-o", "--out_path", dest="out_path", default="", type=str,
                        help="<optional> The ut task result out path.",
                        required=False)
    parser.add_argument('-save_error_data', dest="save_error_data", action="store_true",
                        help="<optional> Save compare failed api output.", required=False)
    parser.add_argument("-j", "--jit_compile", dest="jit_compile", action="store_true",
                        help="<optional> whether to turn on jit compile", required=False)
    parser.add_argument("-d", "--device", dest="device_id", type=int, help="<optional> set device id to run ut",
                        default=0, required=False)
    parser.add_argument("-csv_path", "--result_csv_path", dest="result_csv_path", default="", type=str,
                        help="<optional> The path of accuracy_checking_result_{timestamp}.csv, "
                             "when run ut is interrupted, enter the file path to continue run ut.",
                        required=False)


def _run_ut():
    parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    if not is_gpu:
        torch.npu.set_compile_mode(jit_compile=args.jit_compile)
    used_device = current_device + ":" + str(args.device_id)
    try:
        if is_gpu:
            torch.cuda.set_device(used_device)
        else:
            torch.npu.set_device(used_device)
    except Exception as error:
        print_error_log(f"Set device id failed. device id is: {args.device_id}")
        raise NotImplementedError from error
    check_link(args.forward_input_file)
    forward_file = os.path.realpath(args.forward_input_file)
    check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    save_error_data = args.save_error_data
    forward_content = get_json_contents(forward_file)
    backward_content = {}
    if args.backward_input_file:
        check_link(args.backward_input_file)
        backward_file = os.path.realpath(args.backward_input_file)
        check_file_suffix(backward_file, FileCheckConst.JSON_SUFFIX)
        backward_content = get_json_contents(backward_file)
    result_csv_path = os.path.join(out_path, RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, DETAILS_FILE_NAME)
    test_result_cnt = None
    if args.result_csv_path:
        result_csv_path = get_validated_result_csv_path(args.result_csv_path)
        details_csv_path = get_validated_details_csv_path(result_csv_path)
        test_result_cnt = get_statistics_from_result_csv(result_csv_path)
    if save_error_data:
        if args.result_csv_path:
            time_info = result_csv_path.split('.')[0].split('_')[-1]
            ut_error_data_dir_name = 'ut_error_data' + time_info
            ut_error_data_dir_path = os.path.join(os.path.dirname(result_csv_path), ut_error_data_dir_name)
            global UT_ERROR_DATA_DIR
            UT_ERROR_DATA_DIR = ut_error_data_dir_path
        initialize_save_error_data()
    run_ut_config = RunUTConfig(forward_content, backward_content, result_csv_path, details_csv_path, save_error_data,
                                args.result_csv_path, test_result_cnt)
    run_ut(run_ut_config)


class UtDataInfo:
    def __init__(self, bench_grad_out, device_grad_out, device_out, bench_out, grad_in, in_fwd_data_list):
        self.bench_grad_out = bench_grad_out
        self.device_grad_out = device_grad_out
        self.device_out = device_out
        self.bench_out = bench_out
        self.grad_in = grad_in
        self.in_fwd_data_list = in_fwd_data_list


if __name__ == '__main__':
    _run_ut()
    print_info_log("UT task completed.")
