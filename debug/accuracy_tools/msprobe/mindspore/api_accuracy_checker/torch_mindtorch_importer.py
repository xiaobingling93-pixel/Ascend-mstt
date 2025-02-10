# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

import os
from pathlib import Path

from msprobe.core.common.const import Const, CompareConst, MsCompareConst
import sys
import torch as mindtorch
from torch import Tensor as mindtorch_tensor
import torch.nn.functional as mindtorch_func
import torch_npu as mindtorch_npu
import torch.distributed as mindtorch_dist


is_valid_pt_mt_env = True
is_mt_env = True


def is_mindtorch():
    mindtorch_check_result = False
    try:
        import torch
        from mindspore._c_expression import Tensor
    except ImportError:
        return mindtorch_check_result
    tensor = torch.tensor(0.0)
    if isinstance(tensor, Tensor):
        mindtorch_check_result = True

    return mindtorch_check_result


def remove_torch_related_paths():
    removed_paths = []
    if not is_mindtorch():
        return
    try:
        torch = importlib.import_module("torch")
        torch_file = torch.__file__
    except ImportError:
        return removed_paths

    torch_dir = os.path.dirname(torch_file)

    torch_dir_path = Path(torch_dir).resolve()
    parent_dir = torch_dir_path.parent

    paths_to_remove = [str(parent_dir)]

    for path in paths_to_remove:
        try:
            path_resolved = str(Path(path).resolve())
        except Exception:
            continue

        if path_resolved in sys.path:
            index = sys.path.index(path_resolved)
            removed_paths.append((path_resolved, index))
            sys.path.pop(index)

    return removed_paths


def clear_torch_from_sys_modules():
    modules_to_remove = [module for module in sys.modules if
                         module == "torch" or module.startswith("torch.") or module.startswith(
                             "torch_npu.") or module == "torch_npu"]
    for module in modules_to_remove:
        del sys.modules[module]


def invalid_pt_mt_env():
    global is_valid_pt_mt_env
    is_valid_pt_mt_env = False


def invalid_mt_env():
    global is_mt_env
    is_mt_env = False


def delete_torch_paths():

    removed_paths_total = []
    if not is_mindtorch():
        invalid_pt_mt_env()
        invalid_mt_env()

    clear_torch_from_sys_modules()

    count_delete_env_path = 0
    for count_delete_env_path in range(MsCompareConst.MAX_RECURSION_DEPTH):
        if not is_mindtorch():
            break

        removed = remove_torch_related_paths()
        if removed:
            removed_paths_total.extend(removed)

        clear_torch_from_sys_modules()

    if count_delete_env_path >= MsCompareConst.MAX_RECURSION_DEPTH - 1:
        raise Exception(f"Please check if you have a valid PyTorch and MindTorch environment, and ensure "
                        f"the PYTHONPATH environment variable depth does not exceed {Const.MAX_RECURSION_DEPTH}.")

    return removed_paths_total

initial_sys_path = sys.path.copy()

delete_torch_paths()

import importlib
import gc
gc.collect()
import torch

if is_mindtorch():
    invalid_pt_mt_env()

import torch_npu
import torch.distributed as distributed
from torch import Tensor
import torch.nn.functional as functional

sys.path = initial_sys_path






