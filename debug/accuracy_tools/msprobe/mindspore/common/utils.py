# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
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

import os
import random

import mindspore as ms

from mindspore import ops
from mindspore.mint import nn
from msprobe.core.common.exceptions import DistributedNotInitializedError
from msprobe.core.common.file_utils import path_len_exceeds_limit, check_path_exists, save_npy
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const
from msprobe.core.common.utils import CompareException, check_seed_all


def get_rank_if_initialized():
    if ms.communication.GlobalComm.INITED:
        return ms.communication.get_rank()
    else:
        raise DistributedNotInitializedError("mindspore distributed environment is not initialized")


def convert_bf16_to_fp32(tensor):
    if tensor.dtype == ms.bfloat16:
        tensor = tensor.to(ms.float32)
    return tensor


def save_tensor_as_npy(tensor, file_path):
    if not path_len_exceeds_limit(file_path):
        tensor = convert_bf16_to_fp32(tensor)
        saved_tensor = tensor.asnumpy()
        save_npy(saved_tensor, file_path)
    else:
        logger.warning(f'The file path {file_path} length exceeds limit.')


def convert_to_int(value):
    try:
        return int(value)
    except Exception:
        return -1
    

def clean_input_kwargs(cell):
    if hasattr(cell, 'input_kwargs'):
        del cell.input_kwargs


def list_lowest_level_directories(root_dir):
    check_path_exists(root_dir)
    lowest_level_dirs = []

    def recurse_dirs(current_dir, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error(f'The directory {current_dir} has more than {Const.MAX_DEPTH} levels.')
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        for entry in os.listdir(current_dir):
            full_path = os.path.join(current_dir, entry)
            if os.path.isdir(full_path):
                if any(os.path.isdir(os.path.join(full_path, subentry)) for subentry in os.listdir(full_path)):
                    recurse_dirs(full_path, depth=depth+1)
                else:
                    lowest_level_dirs.append(full_path)

    recurse_dirs(root_dir)
    return lowest_level_dirs


def seed_all(seed=1234, mode=False, rm_dropout=True):
    check_seed_all(seed, mode, rm_dropout)
    os.environ['PYTHONHASHSEED'] = str(seed)
    ms.set_seed(seed)
    random.seed(seed)
    ms.set_context(deterministic="ON" if mode else "OFF")
    os.environ['HCCL_DETERMINISTIC'] = str(mode)
    if rm_dropout:
        remove_dropout()


class MsprobeStep(ms.train.Callback):

    def __init__(self, debugger):
        super(MsprobeStep, self).__init__()
        self.debugger = debugger

    def on_train_step_begin(self, run_context):
        self.debugger.start()

    def on_train_step_end(self, run_context):
        self.debugger.stop()
        self.debugger.step()


class Dropout(ops.Dropout):
    def __init__(self, keep_prob=0.5, Seed0=0, Seed1=1):
        super().__init__(1., Seed0, Seed1)


class Dropout2D(ops.Dropout2D):
    def __init__(self, keep_prob=0.5):
        super().__init__(1.)


class Dropout3D(ops.Dropout3D):
    def __init__(self, keep_prob=0.5):
        super().__init__(1.)


class DropoutExt(nn.Dropout):
    def __init__(self, p=0.5):
        super().__init__(0)


def dropout_ext(input_tensor, p=0.5, training=True):
    return input_tensor


def remove_dropout():
    ops.Dropout = Dropout
    ops.operations.Dropout = Dropout
    ops.Dropout2D = Dropout2D
    ops.operations.Dropout2D = Dropout2D
    ops.Dropout3D = Dropout3D
    ops.operations.Dropout3D = Dropout3D
    nn.Dropout = DropoutExt
    nn.functional.dropout = dropout_ext
