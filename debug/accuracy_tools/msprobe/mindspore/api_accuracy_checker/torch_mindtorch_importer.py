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
import gc
import sys
from pathlib import Path
import mindspore
from msprobe.mindspore.common.log import logger
from msprobe.core.common.const import Const, CompareConst, MsCompareConst
import torch as mindtorch
from torch import Tensor as mindtorch_tensor
import torch.nn.functional as mindtorch_func
import torch.distributed as mindtorch_dist


is_valid_pt_mt_env = True
is_mt_env = True


def is_mindtorch():
    mindtorch_check_result = False
    try:
        import torch as test_torch
        from mindspore._c_expression import Tensor as MindsporeTensor
    except ImportError:
        return mindtorch_check_result
    tensor = test_torch.tensor(0.0)
    if isinstance(tensor, MindsporeTensor):
        mindtorch_check_result = True

    return mindtorch_check_result


def remove_torch_related_paths():
    removed_paths = []
    if not is_mindtorch():
        return removed_paths
    try:
        import torch as remove_torch
        torch_file = remove_torch.__file__
    except ImportError:
        return removed_paths

    torch_dir = os.path.dirname(torch_file)

    torch_dir_path = Path(torch_dir).resolve()
    parent_dir = torch_dir_path.parent

    paths_to_remove = [str(parent_dir)]

    for path in paths_to_remove:
        try:
            path_resolved = str(Path(path).resolve())
        except Exception as error:
            logger.debug(f"Failed to resolve path {path}: {error}")
            continue

        if path_resolved in sys.path:
            index = sys.path.index(path_resolved)
            removed_paths.append((path_resolved, index))
            sys.path.pop(index)

    return removed_paths


def clear_torch_from_sys_modules():
    modules_to_remove = []
    for module in sys.modules:
        if module == "torch" or module.startswith("torch."):
            modules_to_remove.append(module)

    for module in modules_to_remove:
        del sys.modules[module]


def invalid_pt_mt_env():
    global is_valid_pt_mt_env
    is_valid_pt_mt_env = False


def invalid_mt_env():
    global is_mt_env
    is_mt_env = False


def delete_torch_paths():

    if not is_mindtorch():
        invalid_pt_mt_env()
        invalid_mt_env()

    clear_torch_from_sys_modules()

    for count_delete_env_path in range(MsCompareConst.MAX_RECURSION_DEPTH):
        if not is_mindtorch():
            break

        removed = remove_torch_related_paths()

        clear_torch_from_sys_modules()

        if count_delete_env_path >= MsCompareConst.MAX_RECURSION_DEPTH - 1:
            raise Exception(f"Please check if you have a valid PyTorch and MindTorch environment, and ensure "
                            f"the PYTHONPATH environment variable depth does not exceed {Const.MAX_RECURSION_DEPTH}.")


if not is_mindtorch():
    invalid_pt_mt_env()
    invalid_mt_env()

else:
    initial_sys_path = sys.path.copy()
    delete_torch_paths()

    gc.collect()

    import torch

    if is_mindtorch():
        invalid_pt_mt_env()

    sys.path = initial_sys_path


