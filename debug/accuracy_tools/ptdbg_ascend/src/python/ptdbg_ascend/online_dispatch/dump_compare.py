import os
import json
import copy
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from ..common.utils import Const, CompareConst, add_time_as_suffix
from ..compare.acc_compare import cosine_similarity, get_max_abs_err, get_max_relative_err, check_accuracy
from .utils import np_save_data, logger_debug, logger_error, logger_user, COLOR_RED, COLOR_GREEN, COLOR_RESET, \
    CSV_COLUMN_NAME
from ..common.file_check_util import FileOpen, change_mode, FileCheckConst


class DispatchRunParam:
    def __init__(self, debug_flag, device_id, root_npu_path, root_cpu_path, process_num):
        # static parameters are initialized by constructors, and dynamic parameters are constructed at run time
        self.debug_flag = debug_flag
        self.device_id = device_id
        self.root_npu_path = root_npu_path
        self.root_cpu_path = root_cpu_path
        self.process_num = process_num
        self.process_flag = False
        self.func_name = None
        self.func_namespace = None
        self.aten_api = None
        self.aten_api_overload_name = None
        self.single_api_index = None
        self.api_index = None
        self.dump_flag = None
        self.auto_dump_flag = None


class TimeStatistics:
    def __init__(self, name_tag, run_param, timeout=5):
        self.debug = run_param.debug_flag
        if self.debug:
            self.fun = run_param.func_name
            self.device = run_param.device_id
            self.process = run_param.process_num
            self.index = run_param.single_api_index
            self.tag = name_tag
            self.timeout = timeout

    def __enter__(self):
        if self.debug:
            self.time = datetime.now()
            logger_debug(f'Time[{self.tag}]-ENTER: Dev[{self.device}], Pid[{os.getpid()}], Fun[{self.fun}], ' \
                         f'Id[{self.index}]')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.debug:
            cost_time = datetime.now() - self.time
            time_cost = f'Time[{self.tag}]-EXIT: Dev[{self.device}], Pid[{os.getpid()}], Fun[{self.fun}], ' \
                        f'Id[{self.index}], time[{cost_time}]'
            hot_time_cost = "Hotspot " + time_cost

            if cost_time.total_seconds() > self.timeout:
                logger_debug(hot_time_cost)
            else:
                logger_debug(time_cost)


def get_compare_result(npu_data, cpu_data):
    # Do not modify the original data, output delay dump
    if isinstance(npu_data, torch.Tensor):
        npu_npy = npu_data.detach().numpy()
        cpu_npy = cpu_data.detach().numpy()
        # Do not check dtype, there maybe type cast
        if npu_npy.size == 0 or cpu_npy.size == 0:
            return "unsupported", 0, 0, "This is empty data, can not compare."

        if npu_npy.shape != cpu_npy.shape:
            return CompareConst.SHAPE_UNMATCH, CompareConst.SHAPE_UNMATCH, CompareConst.SHAPE_UNMATCH, \
                   "Shape of NPU and bench Tensor do not match. Skipped."

        npu_npy = npu_npy.reshape(-1).astype(float)
        cpu_npy = cpu_npy.reshape(-1).astype(float)
        err_msg = ""
        max_abs_err, _ = get_max_abs_err(npu_npy, cpu_npy)
        max_relative_err, message = get_max_relative_err(npu_npy, cpu_npy)
        if npu_npy.shape == 0:
            return "unsupported", max_abs_err, max_relative_err, "This is type of scalar data, can not compare."

        cos_sim, message = cosine_similarity(npu_npy, cpu_npy)
        err_msg += message

        return cos_sim, max_abs_err, max_relative_err, err_msg
    else:
        npu_npy = np.array(npu_data).astype(float)
        cpu_npy = np.array(cpu_data).astype(float)
        max_abs_err, _ = get_max_abs_err(npu_npy, cpu_npy)
        max_relative_err, _ = get_max_relative_err(npu_npy, cpu_npy)

        return "unsupported", max_abs_err, max_relative_err, "This is type of scalar data, can not compare."


def save_summery(run_param, npu_data, cpu_data, prefix, summery_list, compute_flag):
    data_dict = dict()
    data_dict[CompareConst.NPU_NAME] = prefix
    data_dict[CompareConst.BENCH_NAME] = prefix
    data_dict[CompareConst.NPU_MAX] = []
    data_dict[CompareConst.NPU_MIN] = []
    data_dict[CompareConst.NPU_MEAN] = []
    data_dict[CompareConst.BENCH_MAX] = []
    data_dict[CompareConst.BENCH_MIN] = []
    data_dict[CompareConst.BENCH_MEAN] = []

    if isinstance(npu_data, torch.Tensor) and npu_data.numel() != 0:
        data_dict[CompareConst.NPU_DTYPE] = str(npu_data.dtype)
        data_dict[CompareConst.NPU_SHAPE] = str(list(npu_data.shape))
        data_dict[CompareConst.BENCH_DTYPE] = str(cpu_data.dtype)
        data_dict[CompareConst.BENCH_SHAPE] = str(list(cpu_data.shape))
        # the same process can not call torch api which may capture by torch_dispatch
        if run_param.process_flag:
            data_dict[CompareConst.NPU_MAX] = np.max(npu_data.numpy()).tolist()
            data_dict[CompareConst.NPU_MIN] = np.min(npu_data.numpy()).tolist()
            data_dict[CompareConst.NPU_MEAN] = np.mean(npu_data.numpy()).tolist()
            if compute_flag:
                data_dict[CompareConst.BENCH_MAX] = np.max(cpu_data.numpy()).tolist()
                data_dict[CompareConst.BENCH_MIN] = np.min(cpu_data.numpy()).tolist()
                data_dict[CompareConst.BENCH_MEAN] = np.mean(cpu_data.numpy()).tolist()
            else:
                data_dict[CompareConst.BENCH_MAX] = data_dict[CompareConst.NPU_MAX]
                data_dict[CompareConst.BENCH_MIN] = data_dict[CompareConst.NPU_MIN]
                data_dict[CompareConst.BENCH_MEAN] = data_dict[CompareConst.NPU_MEAN]
    else:
        data_dict[CompareConst.NPU_DTYPE] = str(type(npu_data))
        data_dict[CompareConst.NPU_SHAPE] = str([])
        data_dict[CompareConst.BENCH_DTYPE] = str(type(cpu_data))
        data_dict[CompareConst.BENCH_SHAPE] = str([])
        if run_param.process_flag:
            data_dict[CompareConst.NPU_MAX] = cpu_data
            data_dict[CompareConst.NPU_MIN] = cpu_data
            data_dict[CompareConst.NPU_MEAN] = cpu_data
            data_dict[CompareConst.BENCH_MAX] = cpu_data
            data_dict[CompareConst.BENCH_MIN] = cpu_data
            data_dict[CompareConst.BENCH_MEAN] = cpu_data

    # when need to compute, caller guarantee npu_data and cpu_data is the same type
    if compute_flag:
        data_dict[CompareConst.COSINE], data_dict[CompareConst.MAX_ABS_ERR], data_dict[CompareConst.MAX_RELATIVE_ERR], \
            data_dict[CompareConst.ERROR_MESSAGE] = get_compare_result(npu_data, cpu_data)

        data_dict[CompareConst.ACCURACY] = check_accuracy(data_dict.get(CompareConst.COSINE),
                                                          data_dict.get(CompareConst.MAX_ABS_ERR))
    else:
        data_dict[CompareConst.COSINE] = 1
        data_dict[CompareConst.MAX_ABS_ERR] = 0
        data_dict[CompareConst.MAX_RELATIVE_ERR] = 0
        data_dict[CompareConst.ERROR_MESSAGE] = None
        data_dict[CompareConst.ACCURACY] = CompareConst.ACCURACY_CHECK_YES

    summery_list.append(data_dict)

    if data_dict[CompareConst.ACCURACY] == CompareConst.ACCURACY_CHECK_NO:
        logger_user(f'rank{run_param.device_id} {prefix} index={run_param.single_api_index}, '
                    f'overload={run_param.aten_api_overload_name}, shape={data_dict[CompareConst.NPU_SHAPE]} '
                    f'{COLOR_RED}Failed{COLOR_RESET}, Cosine={data_dict[CompareConst.COSINE]},'
                    f'MaxAbsErr={data_dict[CompareConst.MAX_ABS_ERR]} ')
        return False
    logger_user(f'rank{run_param.device_id} {prefix} index={run_param.single_api_index}, '
                f'shape={data_dict[CompareConst.NPU_SHAPE]} {COLOR_GREEN}Pass{COLOR_RESET}')
    return True


def support_basic_type(data):
    if isinstance(data, (bool, int, float, torch.Tensor)):
        return True
    return False


def dump_data(data, prefix, dump_path):
    if isinstance(data, (tuple, list)) and data:
        for i, item in enumerate(data):
            dump_data(item, "{}.{}".format(prefix, i), dump_path)
        return
    elif support_basic_type(data):
        if isinstance(data, torch.Tensor) and data.is_meta:
            return
        # dump data may greater than summery_list collect
        np_save_data(data, prefix, dump_path)


def compare_data(run_param, npu_data, cpu_data, prefix, summery_list, compute_flag):
    if isinstance(npu_data, (tuple, list)) and npu_data:
        accuracy_reached = True
        for i, (npu_item, cpu_item) in enumerate(zip(npu_data, cpu_data)):
            result = compare_data(run_param, npu_item, cpu_item, "{}.{}".format(prefix, i), summery_list, compute_flag)
            accuracy_reached = accuracy_reached and result
        return accuracy_reached
    elif support_basic_type(npu_data):
        if isinstance(npu_data, torch.Tensor) and npu_data.is_meta:
            return True
        if type(npu_data) != type(cpu_data):
            logger_warn(f'{prefix} can not compare npu type={str(type(npu_data))} cpu type={str(type(cpu_data))}')

            return True
        return save_summery(run_param, npu_data, cpu_data, prefix, summery_list, compute_flag)
    return True


def save_temp_summery(api_index, single_api_summery, path, lock):
    summery_path = os.path.join(path, f'summery.json')
    lock.acquire()
    with FileOpen(summery_path, "a") as f:
        json.dump([api_index, single_api_summery], f)
        f.write('\n')
    lock.release()


def dispatch_workflow(run_param, cpu_args, cpu_kwargs, all_summery, func, npu_out_cpu, cpu_out, lock):
    single_api_summery = []

    prefix_input = f'{run_param.aten_api}_{run_param.single_api_index}_input'
    prefix_output = f'{run_param.aten_api}_{run_param.single_api_index}_output'
    with TimeStatistics("COMPARE INPUT", run_param):
        # assume the input is the same
        compare_data(run_param, cpu_args, cpu_args, prefix_input, single_api_summery, False)
        if len(cpu_kwargs) > 0:
            for k, v in cpu_kwargs.items():
                # kwargs_prefix_name must be the same as the name when dump_data
                kwargs_prefix_name = prefix_input + f'_{k}'
                compare_data(run_param, v, v, kwargs_prefix_name, single_api_summery, False)

    accuracy_reached = False
    with TimeStatistics("COMPARE OUTPUT", run_param):
        accuracy_reached = compare_data(run_param, npu_out_cpu, cpu_out, prefix_output, single_api_summery, True)

    # user set dump or auto mode will dump
    if run_param.dump_flag or (run_param.auto_dump_flag and not accuracy_reached):
        with TimeStatistics("DUMP INPUT", run_param):
            dump_data(cpu_args, prefix_input, run_param.root_npu_path)
            if len(cpu_kwargs) > 0:
                for k, v in cpu_kwargs.items():
                    kwargs_prefix_name = prefix_input + f'_{k}'
                    dump_data(v, kwargs_prefix_name, run_param.root_npu_path)

        with TimeStatistics("DUMP OUTPUT", run_param):
            dump_data(cpu_out, prefix_output, run_param.root_cpu_path)
            dump_data(npu_out_cpu, prefix_output, run_param.root_npu_path)

    if run_param.process_num == 0:
        all_summery[run_param.api_index - 1] = copy.deepcopy(single_api_summery)
    else:
        save_temp_summery(run_param.api_index - 1, single_api_summery, run_param.root_cpu_path, lock)


def get_torch_func(run_param):
    if hasattr(torch.ops, run_param.func_namespace):
        ops_func = getattr(torch.ops, run_param.func_namespace)
        if hasattr(ops_func, run_param.aten_api):
            ops_aten_func = getattr(ops_func, run_param.aten_api)
            if hasattr(ops_aten_func, run_param.aten_api_overload_name):
                ops_aten_overlaod_func = getattr(ops_aten_func, run_param.aten_api_overload_name)
                return ops_aten_overlaod_func
    return None


def dispatch_multiprocess(run_param, cpu_args, cpu_kwargs, all_summery, npu_out_cpu, cpu_out, lock):
    torch_func = get_torch_func(run_param)
    if torch_func is None:
        logger_error(f'can not find suitable call api:{run_param.aten_api}')
    else:
        dispatch_workflow(run_param, cpu_args, cpu_kwargs, all_summery, torch_func, npu_out_cpu, cpu_out, lock)


def error_call(err):
    logger_error(f'multiprocess {err}')


def save_csv(all_summery, call_stack_list, csv_path):
    df = pd.DataFrame(columns=CSV_COLUMN_NAME)

    for index, list_data in enumerate(all_summery):
        for data in list_data:
            csv_row_data = {CompareConst.NPU_NAME: data[CompareConst.NPU_NAME],
                            CompareConst.BENCH_NAME: data[CompareConst.BENCH_NAME],
                            CompareConst.NPU_DTYPE: data[CompareConst.NPU_DTYPE],
                            CompareConst.BENCH_DTYPE: data[CompareConst.BENCH_DTYPE],
                            CompareConst.NPU_SHAPE: data[CompareConst.NPU_SHAPE],
                            CompareConst.BENCH_SHAPE: data[CompareConst.BENCH_SHAPE],
                            CompareConst.NPU_MAX: data[CompareConst.NPU_MAX],
                            CompareConst.NPU_MIN: data[CompareConst.NPU_MIN],
                            CompareConst.NPU_MEAN: data[CompareConst.NPU_MEAN],
                            CompareConst.BENCH_MAX: data[CompareConst.BENCH_MAX],
                            CompareConst.BENCH_MIN: data[CompareConst.BENCH_MIN],
                            CompareConst.BENCH_MEAN: data[CompareConst.BENCH_MEAN],
                            CompareConst.COSINE: data[CompareConst.COSINE],
                            CompareConst.MAX_ABS_ERR: data[CompareConst.MAX_ABS_ERR],
                            CompareConst.MAX_RELATIVE_ERR: data[CompareConst.MAX_RELATIVE_ERR],
                            CompareConst.ACCURACY: data[CompareConst.ACCURACY],
                            CompareConst.STACK: call_stack_list[index],
                            CompareConst.ERROR_MESSAGE: data[CompareConst.ERROR_MESSAGE]}
            row_df = pd.DataFrame.from_dict(csv_row_data, orient='index').T
            df = pd.concat([df, row_df])

    df.to_csv(csv_path, index=False)
    change_mode(csv_path, FileCheckConst.DATA_FILE_AUTHORITY)
