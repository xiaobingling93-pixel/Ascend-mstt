import os
import inspect
import logging
import psutil
import torch
import numpy as np

try:
    import torch_npu
except ImportError:
    pta_cpu_device = None
else:
    pta_cpu_device = torch.device("cpu")

from ..common.utils import print_error_log, CompareConst

cpu_device = torch._C.device("cpu")
COLOR_RED = '\033[31m'
COLOR_GREEN = '\033[32m'
COLOR_YELLOW = '\033[33m'
COLOR_BLUE = '\033[34m'
COLOR_PURPLE = '\033[35m'
COLOR_CYAN = '\033[36m'
COLOR_GRAY = '\033[37m'
COLOR_RESET = '\033[0m'

COMPARE_LOGO = '''
            _ _                                                       
  ___  _ __ | (_)_ __   ___    ___ ___  _ __ ___  _ __   __ _ _ __ ___ 
 / _ \\| '_ \\| | | '_ \\ / _ \\  / __/ _ \\| '_ ` _ \\| '_ \\ / _` | '__/ _ \\ 
| (_) | | | | | | | | |  __/ | (_| (_) | | | | | | |_) | (_| | | |  __/
 \\___/|_| |_|_|_|_| |_|\\___|  \\___\\___/|_| |_| |_| .__/ \\__,_|_|  \\___|
                                                 |_|    
'''

CSV_COLUMN_NAME = [CompareConst.NPU_NAME, 
                   CompareConst.BENCH_NAME,
                   CompareConst.NPU_DTYPE, 
                   CompareConst.BENCH_DTYPE,
                   CompareConst.NPU_SHAPE, 
                   CompareConst.BENCH_SHAPE,
                   CompareConst.NPU_MAX,
                   CompareConst.NPU_MIN,
                   CompareConst.NPU_MEAN,
                   CompareConst.BENCH_MAX,
                   CompareConst.BENCH_MIN,
                   CompareConst.BENCH_MEAN,
                   CompareConst.COSINE, 
                   CompareConst.MAX_ABS_ERR, 
                   CompareConst.MAX_RELATIVE_ERR,
                   CompareConst.ACCURACY, 
                   CompareConst.STACK, 
                   CompareConst.ERROR_MESSAGE]


def get_callstack():
    callstack = []
    for (_, path, line, func, code, _) in inspect.stack()[2:]:
        if code:
            stack_line = [path, str(line), func, code[0].strip() if code else code]
        else:
            stack_line = [path, str(line), func, code]
        callstack.append(stack_line)
    return callstack


def np_save_data(data, file_name, data_path):
    try:
        if hasattr(data, "numpy"):
            data = data.numpy()
        dump_path = os.path.join(data_path, f'{file_name}.npy')
        np.save(dump_path, data)
    except Exception as e:
        print_error_log("save numpy failed, error: {}".format(e))
    finally:
        pass


def data_to_cpu(data, deep, data_cpu):
    global cpu_device
    list_cpu = []
    if isinstance(data, torch.Tensor):
        if data.device == cpu_device or data.device == pta_cpu_device:
            tensor_copy = data.clone().detach()
        else:
            tensor_copy = data.cpu().detach()
        if tensor_copy.dtype in [torch.float16, torch.half, torch.bfloat16]:
            tensor_copy = tensor_copy.float()
        
        if deep == 0:
            data_cpu.append(tensor_copy)
        return tensor_copy
    elif isinstance(data, list):
        for v in data:
            list_cpu.append(data_to_cpu(v, deep + 1, data_cpu))
        if deep == 0:
            data_cpu.append(list_cpu)
        return list_cpu
    elif isinstance(data, tuple):
        for v in data:
            list_cpu.append(data_to_cpu(v, deep + 1, data_cpu))
        tuple_cpu = tuple(list_cpu)
        if deep == 0:
            data_cpu.append(tuple_cpu)
        return tuple_cpu
    elif isinstance(data, dict):
        dict_cpu = {}
        for k, v in data.items():
            dict_cpu[k] = data_to_cpu(v, deep + 1, data_cpu)
        if deep == 0:
            data_cpu.append(dict_cpu)
        return dict_cpu
    elif isinstance(data, torch._C.device):
        return cpu_device
    else:
        return data


def get_mp_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(message)s')
        logger.propagate = True
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger.info


def logger_debug(mesg):
    logger = get_mp_logger()
    logger(f'DEBUG ' + mesg)


def logger_info(mesg):
    logger = get_mp_logger()
    logger(f'INFO ' + mesg)


def logger_warn(mesg):
    logger = get_mp_logger()
    logger(f'{COLOR_YELLOW}WARNING {mesg} {COLOR_RESET}')


def logger_error(mesg):
    logger = get_mp_logger()
    logger(f'{COLOR_RED}ERROR {mesg} {COLOR_RESET}')


def logger_user(mesg):
    logger = get_mp_logger()
    logger(mesg)


def logger_logo():
    logger_user(f'{COLOR_CYAN}{COMPARE_LOGO} {COLOR_RESET}')


def get_sys_info():
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    sys_info = f'Total: {mem.total / 1024 / 1024:.2f}MB '\
               f'Free: {mem.available / 1024 / 1024:.2f} MB '\
               f'Used: {mem.used / 1024 / 1024:.2f} MB '\
               f'CPU: {cpu_percent}% '
    return sys_info
