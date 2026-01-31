# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


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
