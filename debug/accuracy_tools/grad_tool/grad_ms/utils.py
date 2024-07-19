import os

import numpy as np

from grad_tool.common.constant import GradConst
from grad_tool.common.utils import print_warn_log, create_directory, change_mode

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
    grad_direction_tensor = grad > 0
    grad_direction_ndarray = grad_direction_tensor.numpy()

    save_filepath = os.path.join(save_path, f"{param_name}.npy")
    np.save(save_filepath, grad_direction_ndarray)
    change_mode(save_filepath, 0o640)

def get_adapted_level(level: str):
    if level == GradConst.LEVEL3:
        print_warn_log(f"In mindpsore pynative mode, only 'L0', 'L1' and 'L2' are supported, use L0 instead")
        level = GradConst.LEVEL0
    level_adapted = level_adp.get(level)
    return level_adapted