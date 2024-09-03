import os

import mindspore
from msprobe.core.grad_probe.constant import level_adp
from msprobe.core.grad_probe.utils import check_param
from msprobe.core.common.file_utils import (create_directory,
                                            check_path_before_create,
                                            check_file_or_directory_path,
                                            save_npy)


def save_grad_direction(param_name, grad, save_path):
    if not os.path.exists(save_path):
        create_directory(save_path)
    check_file_or_directory_path(save_path, isdir=True)
    check_param(param_name)
    save_filepath = os.path.join(save_path, f"{param_name}.npy")
    check_path_before_create(save_filepath)

    if grad.dtype == mindspore.bfloat16:
        grad = grad.to(mindspore.float32)
    grad_direction_tensor = grad > 0
    grad_direction_ndarray = grad_direction_tensor.numpy()

    save_npy(grad_direction_ndarray, save_filepath)


def get_adapted_level(level: str):
    level_adapted = level_adp.get(level)
    return level_adapted