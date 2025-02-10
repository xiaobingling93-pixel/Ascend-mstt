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


import importlib
import inspect

print(f"\ mindtorch_module  mindtorch_npu.__file__ 重新验证: {mindtorch_npu.__file__}")

module_name = mindtorch_tensor.__module__
file_path = inspect.getfile(importlib.import_module(module_name))
print(f"mindtorch_module Module {module_name} file path: {file_path}")



# 判断是否使用 MindTorch
def is_mindtorch():
    mindtorch_check_result = False
    try:
        # clear_torch_from_sys_modules()
        # import torch
        # torch = importlib.reload(torch)
        torch = importlib.import_module("torch")
        # clear_torch_from_sys_modules()
        from mindspore._c_expression import Tensor
    except ImportError:
        return mindtorch_check_result
    tensor = torch.tensor(0.0)
    if isinstance(tensor, Tensor):
        mindtorch_check_result = True

    return mindtorch_check_result


# 删除与 torch 相关的路径
def remove_torch_related_paths():
    removed_paths = []
    if not is_mindtorch():
        return
    try:
        torch = importlib.import_module("torch")
        torch_file = torch.__file__
        print(f"torch.__file__: {torch_file}")
    except ImportError:
        print("无法获取 torch.__file__。请确保 torch 已安装。")
        return removed_paths

    torch_dir = os.path.dirname(torch_file)

    torch_dir_path = Path(torch_dir).resolve()
    parent_dir = torch_dir_path.parent

    # paths_to_remove = str(parent_dir)
    #
    # try:
    #     path_resolved = str(Path(paths_to_remove).resolve())
    # except Exception:
    #     print(f"无法解析路径 '{paths_to_remove}'，跳过。")
    #
    # if path_resolved in sys.path:
    #     index = sys.path.index(path_resolved)
    #     removed_paths.append((path_resolved, index))
    #     sys.path.pop(index)

    paths_to_remove = [str(parent_dir)]

    for path in paths_to_remove:
        try:
            path_resolved = str(Path(path).resolve())
        except Exception:
            print(f"无法解析路径 '{path}'，跳过。")
            continue

        if path_resolved in sys.path:
            index = sys.path.index(path_resolved)
            removed_paths.append((path_resolved, index))
            sys.path.pop(index)
            print(f"已从 sys.path 中删除 '{path_resolved}' (索引: {index})")
        else:
            print(f"路径 '{path_resolved}' 不在 sys.path 中，无需删除。")

    return removed_paths


def clear_torch_from_sys_modules():
    modules_to_remove = [module for module in sys.modules if
                         module == "torch" or module.startswith("torch.") or module.startswith(
                             "torch_npu.") or module == "torch_npu"]
    for module in modules_to_remove:
        del sys.modules[module]


is_valid_pt_mt_env = True
is_mt_env = True

def invalid_pt_mt_env():
    global is_valid_pt_mt_env
    is_valid_pt_mt_env = False

def invalid_mt_env():
    global is_mt_env
    is_mt_env = False
# 新增函数：删除 torch 路径的主流程
def delete_torch_paths():
    print("初始 sys.path:")
    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")

    removed_paths_total = []
    if not is_mindtorch():
        invalid_pt_mt_env()
        invalid_mt_env()

    clear_torch_from_sys_modules()

    count_delete_env_path = 0
    for count_delete_env_path in range(MsCompareConst.MAX_RECURSION_DEPTH):
        print("\n开始删除与 torch 相关的路径...\n")
        if is_mindtorch():
            print("\nmindtorch 仍然可导入，继续删除相关路径...")
        else:
            print("\nmindtorch 无法导入，停止删除。")
            break

        removed = remove_torch_related_paths()
        if removed:
            removed_paths_total.extend(removed)
            print("\n已删除的路径:")
            for path, index in removed:
                print(f"'{path}' (索引: {index})")
        else:
            print("\n没有找到需要删除的路径。")

        print("\n更新后的 sys.path:")
        for i, path in enumerate(sys.path):
            print(f"{i}: {path}")
        clear_torch_from_sys_modules()

    if count_delete_env_path >= MsCompareConst.MAX_RECURSION_DEPTH - 1:
        raise Exception(f"Please check if you have a valid PyTorch and MindTorch environment, and ensure "
                        f"the PYTHONPATH environment variable depth does not exceed {Const.MAX_RECURSION_DEPTH}.")


    # clear_torch_from_sys_modules()

    return removed_paths_total


print(f"开始sys.path: {sys.path}")
initial_sys_path = sys.path.copy()

delete_torch_paths()

# clear_torch_from_sys_modules()
# print(sys.modules)
# print(f"\ntorch.__file__ 重新验证走进没api_runner: {torch.__file__}")

import importlib
import gc
gc.collect()
# torch = importlib.import_module("torch")
import torch

if is_mindtorch():
    invalid_pt_mt_env()
import torch_npu
import torch.distributed as distributed
from torch import Tensor
import torch.nn.functional as functional
print(f"\n限制递归深度并且当不存在pytorch和mindtorch报错: {torch.__file__}")
print(f"\ntorch.__file__ 重新验证走进没api_runner: {torch.__file__}")

print(f"\ntorch.__file__ 重新验证: {mindtorch.__file__}")
print(f"\ntorch.__file__ 重新验证: {torch_npu.__file__}")
print(f"\ntorch.__file__ 重新验证: {mindtorch_npu.__file__}")
sys.path = initial_sys_path
print(f"已恢复 sys.path: {sys.path}")





