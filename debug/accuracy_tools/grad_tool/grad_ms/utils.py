import os

import numpy as np
import mindspore
from grad_tool.common.constant import GradConst
from grad_tool.common.utils import (print_warn_log, create_directory, change_mode, check_file_or_directory_path,
                                    path_valid_check, check_param)

level_adp = {
        "L0": {
            "header": [GradConst.MD5, GradConst.MAX, GradConst.MIN, GradConst.NORM, GradConst.SHAPE],
            "have_grad_direction": False
        },
        "L1": {
            "header": [GradConst.MAX, GradConst.MIN, GradConst.NORM, GradConst.SHAPE],
            "have_grad_direction": True
        },
        "L2": {
            "header": [GradConst.DISTRIBUTION, GradConst.MAX, GradConst.MIN, GradConst.NORM, GradConst.SHAPE],
            "have_grad_direction": True
        },
    }


def save_grad_direction(param_name, grad, save_path):
    if not os.path.exists(save_path):
        create_directory(save_path)
    check_file_or_directory_path(save_path, file_type=GradConst.DIR)
    check_param(param_name)
    save_filepath = os.path.join(save_path, f"{param_name}.npy")
    path_valid_check(save_filepath)

    if grad.dtype == mindspore.bfloat16:
        grad = grad.to(mindspore.float32)
    grad_direction_tensor = grad > 0
    grad_direction_ndarray = grad_direction_tensor.numpy()

    try:
        np.save(save_filepath, grad_direction_ndarray)
    except Exception as e:
        raise RuntimeError("An unexpected error occurred: %s when saving numpy to %s" % (str(e), save_filepath))
    change_mode(save_filepath, 0o640)


def get_adapted_level(level: str):
    level_adapted = level_adp.get(level)
    return level_adapted
