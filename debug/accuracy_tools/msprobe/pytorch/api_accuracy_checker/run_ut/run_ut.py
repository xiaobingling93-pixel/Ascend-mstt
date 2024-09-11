import argparse
import os
import csv
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

from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import Backward_Message, hf_32_standard_api, UtDataInfo, \
    get_validated_result_csv_path, get_validated_details_csv_path, exec_api
from msprobe.pytorch.api_accuracy_checker.run_ut.data_generate import gen_api_params, gen_args
from msprobe.pytorch.api_accuracy_checker.common.utils import api_info_preprocess, \
    initialize_save_path, UtDataProcessor, extract_basic_api_segments, ApiData
from msprobe.pytorch.api_accuracy_checker.compare.compare import Comparator
from msprobe.pytorch.api_accuracy_checker.compare.compare_column import CompareColumn
from msprobe.pytorch.api_accuracy_checker.common.config import msCheckerConfig
from msprobe.pytorch.common.parse_json import parse_json_info_forward_backward
from msprobe.core.common.file_utils import FileOpen, FileChecker, \
    change_mode, check_path_before_create, create_directory, get_json_contents
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.pt_config import parse_json_config
from msprobe.core.common.const import Const, FileCheckConst, CompareConst
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.attl import ATTL, ATTLConfig, move2device_exec
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.device_dispatch import ConsumerDispatcher


current_time = time.strftime("%Y%m%d%H%M%S")
UT_ERROR_DATA_DIR = 'ut_error_data' + current_time
RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + ".csv"
RunUTConfig = namedtuple('RunUTConfig', ['forward_content', 'backward_content', 'result_csv_path', 'details_csv_path',
                                         'save_error_data', 'is_continue_run_ut', 'real_data_path', 'white_list',
                                         'black_list', 'error_data_path', 'online_config'])

OnlineConfig = namedtuple('OnlineConfig', ['is_online', 'nfs_path', 'host', 'port', 'rank_list', 'tls_path'])

not_backward_list = ['repeat_interleave']
not_detach_set = {'resize_', 'resize_as_', 'set_', 'transpose_', 't_', 'squeeze_', 'unsqueeze_'}
not_raise_dtype_set = {'type_as'}

RAISE_PRECISION = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float64
}

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


def deal_detach(arg, to_detach=True):
    return arg.detach() if to_detach else arg


def raise_bench_data_dtype(api_name, arg, raise_dtype=None):
    '''
    将标杆数据的dtype转换为raise_dtype
    输入：
        api_name：api名称
        arg：标杆输入
        raise_dtype：需要转换的dtype
    输出： 
        arg: 转换dtype的标杆输入
    '''
    if api_name in hf_32_standard_api and arg.dtype == torch.float32:
        return arg
    if raise_dtype is None or arg.dtype not in RAISE_PRECISION or raise_dtype == arg.dtype:
        return arg
    return arg.type(raise_dtype)


def generate_device_params(input_args, input_kwargs, need_backward, api_name):
    def recursive_arg_to_device(arg_in, to_detach):
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

    is_detach = api_name not in not_detach_set
    device_args = recursive_arg_to_device(input_args, is_detach)
    device_kwargs = \
        {key: recursive_arg_to_device(value, key != "out" and is_detach) for key, value in input_kwargs.items()}
    return device_args, device_kwargs


def generate_cpu_params(input_args, input_kwargs, need_backward, api_name):
    def recursive_arg_to_cpu(arg_in, to_detach, raise_dtype=None):
        if isinstance(arg_in, (list, tuple)):
            return type(arg_in)(recursive_arg_to_cpu(arg, to_detach, raise_dtype=raise_dtype) for arg in arg_in)
        elif isinstance(arg_in, torch.Tensor):
            if need_backward and arg_in.requires_grad:
                arg_in = deal_detach(raise_bench_data_dtype(
                                     api_name, arg_in.clone(), raise_dtype=raise_dtype), to_detach).requires_grad_()
                temp_arg_in = arg_in * 1
                arg_in = temp_arg_in.type_as(arg_in)
                arg_in.retain_grad()
                return arg_in
            else:
                return deal_detach(raise_bench_data_dtype(api_name, arg_in.clone(), raise_dtype=raise_dtype), to_detach)
        else:
            return arg_in

    def is_tensor_with_raise_precision(arg_in, check_kwargs=False):
        if arg_in.dtype in RAISE_PRECISION:
            return True
        if check_kwargs and arg_in.dtype in [torch.half, torch.bfloat16]:
            return True
        return False

    def recursive_find_dtypes(arg_in, kwargs=None, check_kwargs=False):
        if isinstance(arg_in, (list, tuple)):
            return set().union(*tuple(recursive_find_dtypes(arg, kwargs, check_kwargs=check_kwargs) for arg in arg_in))
        elif isinstance(arg_in, torch.Tensor) and is_tensor_with_raise_precision(arg_in, check_kwargs):
            return set([arg_in.dtype])
        elif isinstance(arg_in, dict) and check_kwargs:
            return set().union(*tuple(recursive_find_dtypes(v, kwargs, check_kwargs=True) for v in arg_in.values()))
        return set()

    raise_dtype = None
    need_raise_dtypes = recursive_find_dtypes(input_args)
    need_raise_dtypes.update(recursive_find_dtypes(input_kwargs, check_kwargs=True))
    if len(need_raise_dtypes) == 1:
        raise_dtype = RAISE_PRECISION.get(need_raise_dtypes.pop(), torch.float32)
    elif len(need_raise_dtypes) >= 2:
        raise_dtype = torch.float32

    raise_dtype = None if api_name in not_raise_dtype_set else raise_dtype
    is_detach = api_name not in not_detach_set
    cpu_args = recursive_arg_to_cpu(input_args, is_detach, raise_dtype=raise_dtype)
    cpu_kwargs = {key: recursive_arg_to_cpu(value, key != "out" and is_detach, raise_dtype=raise_dtype) for key, value in input_kwargs.items()}
    return cpu_args, cpu_kwargs


def run_ut(config):
    logger.info("start UT test")
    if config.online_config.is_online:
        logger.info(f"UT task result will be saved in {config.result_csv_path}".replace(".csv", "_rank*.csv"))
        logger.info(f"UT task details will be saved in {config.details_csv_path}".replace(".csv", "_rank*.csv"))
    else:
        logger.info(f"UT task result will be saved in {config.result_csv_path}")
        logger.info(f"UT task details will be saved in {config.details_csv_path}")

    if config.save_error_data:
        logger.info(f"UT task error_datas will be saved in {config.error_data_path}")
    compare = Comparator(config.result_csv_path, config.details_csv_path, config.is_continue_run_ut, config=config)

    if config.online_config.is_online:
        run_api_online(config, compare)
    else:
        with FileOpen(config.result_csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            api_name_set = {row[0] for row in csv_reader}
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
        if api_full_name in api_name_set:
            continue
        if is_unsupported_api(api_full_name):
            continue
        _, api_name = extract_basic_api_segments(api_full_name)
        if not api_name:
            err_message = f"API {api_full_name} not support for run ut. SKIP."
            logger.error(err_message)
            fwd_compare_alg_results = err_column.to_column_value(CompareConst.SKIP, err_message)
            result_info = (api_full_name, CompareConst.SKIP, CompareConst.SKIP, [fwd_compare_alg_results], None, 0)
            compare.record_results(result_info)
            continue
        try:
            if blacklist_and_whitelist_filter(api_name, config.black_list, config.white_list):
                continue
            data_info = run_torch_api(api_full_name, config.real_data_path, config.backward_content, api_info_dict)
            is_fwd_success, is_bwd_success = compare.compare_output(api_full_name, data_info)
            if config.save_error_data:
                do_save_error_data(api_full_name, data_info, config.error_data_path, is_fwd_success, is_bwd_success)
        except Exception as err:
            if "expected scalar type Long" in str(err):
                logger.warning(f"API {api_name} not support int32 tensor in CPU, please add {api_name} to CONVERT_API "
                               f"'int32_to_int64' list in accuracy_tools/api_accuracy_check/common/utils.py file.")
            else:
                logger.error(f"Run {api_full_name} UT Error: %s" % str(err))
            fwd_compare_alg_results = err_column.to_column_value(CompareConst.SKIP, str(err))
            result_info = (api_full_name, CompareConst.SKIP, CompareConst.SKIP, [fwd_compare_alg_results], None, 0)
            compare.record_results(result_info)
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
    if black_list and api_name in black_list:
        return True
    if white_list and api_name not in white_list:
        return True
    return False


def is_unsupported_api(api_name):
    split_name = api_name.split(Const.SEP)[0]
    flag = split_name == Const.DISTRIBUTED
    if flag:
        logger.info(f"{split_name} api is not supported for run ut. SKIP.")
    return flag


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
    args, kwargs, need_grad = get_api_info(api_info_dict, api_name, real_data_path)
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    need_backward = api_full_name in backward_content
    if not need_grad:
        logger.warning("%s %s" % (api_full_name, Backward_Message.UNSUPPORT_BACKWARD_MESSAGE))
        backward_message += Backward_Message.UNSUPPORT_BACKWARD_MESSAGE
    if api_name in not_backward_list:
        need_grad = False
        logger.warning("%s %s" % (api_full_name, Backward_Message.NO_BACKWARD_RESULT_MESSAGE))
        backward_message += Backward_Message.NO_BACKWARD_RESULT_MESSAGE
    need_backward = need_backward and need_grad
    if kwargs.get("device"):
        del kwargs["device"]
    cpu_args, cpu_kwargs = generate_cpu_params(args, kwargs, need_backward, api_name)
    device_args, device_kwargs = generate_device_params(args, kwargs, need_backward, api_name)
    bench_grad_out, device_grad_out = None, None
    out = exec_api(api_type, api_name, Const.CPU_LOWERCASE, cpu_args, cpu_kwargs)
    device_out = exec_api(api_type, api_name, current_device, device_args, device_kwargs)
    current_path = os.path.dirname(os.path.realpath(__file__))
    ut_setting_path = os.path.join(current_path, "torch_ut_setting.json")
    api_setting_dict = get_json_contents(ut_setting_path)
    grad_input_index = api_setting_dict.get(api_name)
    grad_index = None
    grad, bench_grad = None, None
    if grad_input_index is not None:
        grad_index = grad_input_index.get('grad_index')

    if need_backward:
        if need_to_backward(grad_index, out):
            backward_args = backward_content[api_full_name].get("input")
            grad = gen_args(backward_args, api_name, real_data_path=real_data_path)[0]
            bench_grad, _ = generate_cpu_params(grad, {}, False, api_name)
            bench_grad_out = run_backward(cpu_args, bench_grad, grad_index, out)
            device_grad = grad.clone().detach().to(current_device)
            device_grad_out = run_backward(device_args, device_grad, grad_index, device_out)
        else:
            backward_message += Backward_Message.MULTIPLE_BACKWARD_MESSAGE
    if api_name == "npu_fusion_attention":
        out = out[0]
        device_out = device_out[0]

    return UtDataInfo(bench_grad_out, device_grad_out, device_out, out, bench_grad, in_fwd_data_list, backward_message)


def run_torch_api_online(api_full_name, api_data, backward_content):
    in_fwd_data_list = []
    api_type, api_name = extract_basic_api_segments(api_full_name)
    args, kwargs, out = api_data.args, api_data.kwargs, api_data.result
    in_fwd_data_list.append(args)
    in_fwd_data_list.append(kwargs)
    if kwargs.get("device"):
        del kwargs["device"]

    device_out = exec_api(api_type, api_name, Const.CUDA_LOWERCASE, args, kwargs)
    device_out = move2device_exec(device_out, "cpu")
    return UtDataInfo(None, None, out, device_out, None, in_fwd_data_list, None, rank=api_data.rank)


def get_api_info(api_info_dict, api_name, real_data_path):
    convert_type, api_info_dict = api_info_preprocess(api_name, api_info_dict)
    need_grad = True
    if api_info_dict.get("input_kwargs") and "out" in api_info_dict.get("input_kwargs"):
        need_grad = False
    args, kwargs = gen_api_params(api_info_dict, api_name, need_grad, convert_type, real_data_path)
    return args, kwargs, need_grad


def need_to_backward(grad_index, out):
    if grad_index is None and isinstance(out, (list, tuple)):
        return False
    return True


def run_backward(args, grad, grad_index, out):
    if grad_index is not None:
        out[grad_index].backward(grad)
    else:
        out.backward(grad)
    args_grad = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            args_grad.append(arg.grad)
    grad_out = args_grad

    return grad_out


def initialize_save_error_data(error_data_path):
    check_path_before_create(error_data_path)
    create_directory(error_data_path)
    error_data_path_checker = FileChecker(error_data_path, FileCheckConst.DIR,
                                          ability=FileCheckConst.WRITE_ABLE)
    error_data_path = error_data_path_checker.common_check()
    error_data_path =initialize_save_path(error_data_path, UT_ERROR_DATA_DIR)
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
        base_key = key.rsplit(Const.SEP, 1)[0]

        if key not in arg_cache:
            filtered_new_args = [
                {k: v for k, v in arg.items() if k not in ['Max', 'Min']}
                for arg in value['input_args'] if isinstance(arg, dict)
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


def run_ut_command(args):
    if not is_gpu:
        torch.npu.set_compile_mode(jit_compile=args.jit_compile)
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
        api_info_file_checker = FileChecker(file_path = args.api_info_file, path_type = FileCheckConst.FILE, 
                                            ability = FileCheckConst.READ_ABLE, file_type = FileCheckConst.JSON_SUFFIX)
        checked_api_info = api_info_file_checker.common_check()
        forward_content, backward_content, real_data_path = parse_json_info_forward_backward(checked_api_info)
        if args.filter_api:
            logger.info("Start filtering the api in the forward_input_file.")
            forward_content = preprocess_forward_content(forward_content)
            logger.info("Finish filtering the api in the forward_input_file.")

    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    check_path_before_create(out_path)
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    save_error_data = args.save_error_data

    result_csv_path = os.path.join(out_path, RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, DETAILS_FILE_NAME)
    if args.result_csv_path:
        result_csv_path = get_validated_result_csv_path(args.result_csv_path, 'result')
        details_csv_path = get_validated_details_csv_path(result_csv_path)
    white_list = msCheckerConfig.white_list
    black_list = msCheckerConfig.black_list
    error_data_path = msCheckerConfig.error_data_path
    is_online = msCheckerConfig.is_online
    nfs_path = msCheckerConfig.nfs_path
    host = msCheckerConfig.host
    port = msCheckerConfig.port
    rank_list = msCheckerConfig.rank_list
    tls_path = msCheckerConfig.tls_path
    if args.config_path:
        config_path_checker = FileChecker(args.config_path, FileCheckConst.FILE, 
                                          FileCheckConst.READ_ABLE, FileCheckConst.JSON_SUFFIX)
        checked_config_path = config_path_checker.common_check()
        _, task_config = parse_json_config(checked_config_path, Const.RUN_UT)
        white_list = task_config.white_list
        black_list = task_config.black_list
        error_data_path = task_config.error_data_path
        is_online = task_config.is_online
        nfs_path = task_config.nfs_path
        host = task_config.host
        port = task_config.port
        rank_list = task_config.rank_list
        tls_path = task_config.tls_path

    if save_error_data:
        if args.result_csv_path:
            time_info = result_csv_path.split('.')[0].split('_')[-1]
            global UT_ERROR_DATA_DIR
            UT_ERROR_DATA_DIR = 'ut_error_data' + time_info
        error_data_path = initialize_save_error_data(error_data_path)
    online_config = OnlineConfig(is_online, nfs_path, host, port, rank_list, tls_path)
    run_ut_config = RunUTConfig(forward_content, backward_content, result_csv_path, details_csv_path, save_error_data,
                                args.result_csv_path, real_data_path, set(white_list), set(black_list), error_data_path,
                                online_config)
    run_ut(run_ut_config)


if __name__ == '__main__':
    _run_ut()
    logger.info("UT task completed.")
