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

import io
import os
import random
import stat
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist
from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.core.common.file_utils import (FileCheckConst, change_mode,
                                            check_file_or_directory_path, check_path_before_create)
from msprobe.core.common.log import logger
from msprobe.core.common.utils import check_seed_all
from packaging import version

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

torch_without_guard_version = torch.__version__ >= '2.1'

if not is_gpu and not torch_without_guard_version:
    from torch_npu.utils.device_guard import torch_device_guard as torch_npu_device_guard

npu_distributed_api = ['isend', 'irecv']


def parameter_adapter(func):
    def handle_masked_select(input_tensor, indices):
        masked_select_func = getattr(torch._C._VariableFunctionsClass, "masked_select")
        if input_tensor.dtype == torch.bfloat16:
            # masked_select在NPU上输入数据dtype类型为bfloat16会报错，提示不支持此类型
            return masked_select_func(input_tensor.to(torch.float32), indices).to(torch.bfloat16)
        else:
            return masked_select_func(input_tensor, indices)

    @wraps(func)
    def inner(self, *args, **kwargs):
        if self.op_name_ == "__getitem__" and len(args) > 1 and isinstance(args[1], torch.Tensor):
            input_tensor = args[0]
            indices = args[1]
            if indices.dtype == torch.uint8:
                indices = indices.bool()
            if indices.dtype == torch.bool:
                if indices.shape == input_tensor.shape:
                    return handle_masked_select(input_tensor, indices)
                else:
                    indices = getattr(torch._C._VariableFunctionsClass, "nonzero")(indices, as_tuple=True)
                    return getattr(torch._C._TensorBase, "__getitem__")(input_tensor, indices)
            elif indices.dtype != torch.bool:
                if not indices.shape or len(indices.shape) == 1:
                    return func(self, input_tensor, indices.tolist())
                elif len(indices.shape) == 2:
                    result = [func(self, input_tensor, index) for index in indices.tolist()]
                    return getattr(torch._C._VariableFunctionsClass, "stack")(result, 0)
                else:
                    res = [input_tensor[tensor_index] for tensor_index in indices]
                    return getattr(torch._C._VariableFunctionsClass, "stack")(res, 0)
        if self.op_name_ == "__eq__" and args[1] is None:
            return False
        return func(self, *args, **kwargs)

    return inner


def torch_device_guard(func):
    if is_gpu or torch_without_guard_version:
        return func

    # Parse args/kwargs matched torch.device objects
    @torch_npu_device_guard
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def get_rank_if_initialized():
    """
        return rank id if it is initialized or raise Exception: DistributedNotInitializedError
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        raise DistributedNotInitializedError("torch distributed environment is not initialized")


def seed_all(seed=1234, mode=False):
    check_seed_all(seed, mode)
    try:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cuda_version = torch.version.cuda
        if cuda_version is not None and version.parse(cuda_version) >= version.parse("10.2"):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['HCCL_DETERMINISTIC'] = str(mode)
        torch.use_deterministic_algorithms(mode)
        if is_gpu:
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enable = False
            torch.backends.cudnn.benchmark = False
        else:
            torch_npu.npu.manual_seed_all(seed)
            torch_npu.npu.manual_seed(seed)
    except Exception as e:
        logger.error(f"There is an unexpected error while determinating randomness. {e}")


class Const:
    """
    Class for const
    """
    SEP = "."
    MODEL_TYPE = ['.onnx', '.pb', '.om']
    DIM_PATTERN = r"^(-?[0-9]+)(,-?[0-9]+)*"
    SEMICOLON = ";"
    COLON = ":"
    EQUAL = "="
    COMMA = ","
    DOT = "."
    DUMP_RATIO_MAX = 100
    SUMMERY_DATA_NUMS = 256
    FLOAT_EPSILON = np.finfo(float).eps
    SUPPORT_DUMP_MODE = ['api', 'acl']
    ON = 'ON'
    OFF = 'OFF'
    KWARGS = 'kwargs'
    INPUT = 'input'
    OUTPUT = 'output'
    BACKWARD = 'backward'
    FORWARD = 'forward'
    PRE_FORWARD = "pre_forward"
    INPUT_ARGS = 'input_args'
    INPUT_KWARGS = 'input_kwargs'
    GRAD_INPUT = 'grad_input'
    GRAD_OUTPUT = 'grad_output'
    START = "start"
    STOP = "stop"
    MAX = 'Max'
    MIN = 'Min'

    # dump mode
    ALL = "all"
    LIST = "list"
    RANGE = "range"
    STACK = "stack"
    ACL = "acl"
    API_LIST = "api_list"
    API_STACK = "api_stack"
    DUMP_MODE = [ALL, LIST, RANGE, STACK, ACL, API_LIST, API_STACK]
    AUTO = "auto"
    ONLINE_DUMP_MODE = [ALL, LIST, AUTO, OFF]
    SUMMARY = "summary"
    MD5 = "md5"
    SUMMARY_MODE = [ALL, SUMMARY, MD5]

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    OVERWRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR

    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    ONE_GB = 1 * 1024 * 1024 * 1024
    TEN_GB = 10 * 1024 * 1024 * 1024
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    FILE_NAME_LENGTH = 255
    DIRECTORY_LENGTH = 4096
    DISTRIBUTED_PREFIX_LENGTH = 60
    SUMMARY_COLUMN_NUM = 6
    STACK_COLUMN_NUM = 2
    # env dump path
    ASCEND_WORK_PATH = "ASCEND_WORK_PATH"
    DUMP_DIR = "dump_data"
    DATA = "data"

    ENV_ENABLE = "1"
    ENV_DISABLE = "0"

    MAX_SEED_VALUE = 2 ** 32 - 1

    TASK_LIST = ["tensor", "statistics", "overflow_check", "free_benchmark"]
    LEVEL_LIST = ["L0", "L1", "L2", "mix"]
    STATISTICS = "statistics"
    TENSOR = "tensor"
    OVERFLOW_CHECK = "overflow_check"
    FREE_BENCHMARK = "free_benchmark"

    ATTR_NAME_PREFIX = "wrap_"

    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble, np.float32, np.float16]
    BOOL_TYPE = [bool, np.uint8]
    INT_TYPE = [np.int32, np.int64]
    NPU = 'NPU'
    DISTRIBUTED = 'Distributed'

    RAISE_PRECISION = {
        torch.float16: torch.float32,
        torch.bfloat16: torch.float32,
        torch.float32: torch.float64
    }
    CONVERT = {
        "int32_to_int64": ["torch.int32", "torch.int64"],
    }

    CONVERT_API = {
        "int32_to_int64": ["cross_entropy"]
    }


def get_tensor_rank(in_feat, out_feat):
    if dist.is_initialized():
        return dist.get_rank()

    def get_tensor_rank_single(x):
        if isinstance(x, (list, tuple)):
            if len(x) > 0:
                return get_tensor_rank_single(x[0])
        elif isinstance(x, torch.Tensor):
            device = x.device
            if device.type != 'cpu':
                return device.index
        return None

    in_rank = get_tensor_rank_single(in_feat)
    out_rank = get_tensor_rank_single(out_feat)
    tensor_rank = in_rank if in_rank else out_rank
    return tensor_rank


def get_rank_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def print_rank_0(message):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logger.info(message)
    else:
        logger.info(message)


def load_pt(pt_path, to_cpu=False):
    pt_path = os.path.realpath(pt_path)
    check_file_or_directory_path(pt_path)
    try:
        if to_cpu:
            pt = torch.load(pt_path, map_location=torch.device("cpu"))
        else:
            pt = torch.load(pt_path)
    except Exception as e:
        raise RuntimeError(f"load pt file {pt_path} failed") from e
    return pt


def save_pt(tensor, filepath):
    filepath = os.path.realpath(filepath)
    check_path_before_create(filepath)
    try:
        torch.save(tensor, filepath)
    except Exception as e:
        logger.error("Save pt file failed, please check according possible error causes: "
                     "1. out of disk space or disk error, "
                     "2. no permission to write files, etc.")
        raise RuntimeError(f"save pt file {filepath} failed") from e
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


def save_api_data(api_data):
    """Save data to io stream"""
    try:
        io_buff = io.BytesIO()
        torch.save(api_data, io_buff)
    except Exception as e:
        raise RuntimeError(f"save api_data to io_buff failed") from e
    return io_buff


def load_api_data(api_data_bytes):
    """Load data from bytes stream"""
    try:
        buffer = io.BytesIO(api_data_bytes)
        buffer = torch.load(buffer, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"load api_data from bytes failed") from e
    return buffer
