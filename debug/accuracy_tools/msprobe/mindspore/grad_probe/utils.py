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

import os

import mindspore
from msprobe.core.common.file_utils import (create_directory,
                                            check_file_or_directory_path,
                                            save_npy)
from msprobe.core.grad_probe.constant import level_adp
from msprobe.core.grad_probe.utils import check_param


def save_grad_direction(param_name, grad, save_path):
    if not os.path.exists(save_path):
        create_directory(save_path)
    check_file_or_directory_path(save_path, isdir=True)
    check_param(param_name)
    save_filepath = os.path.join(save_path, f"{param_name}.npy")

    if grad.dtype == mindspore.bfloat16:
        grad = grad.to(mindspore.float32)
    grad_direction_tensor = grad > 0
    grad_direction_ndarray = grad_direction_tensor.numpy()

    save_npy(grad_direction_ndarray, save_filepath)


def get_adapted_level(level: str):
    level_adapted = level_adp.get(level)
    return level_adapted
