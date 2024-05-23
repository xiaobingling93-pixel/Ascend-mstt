from abc import ABC, abstractmethod

import torch

from .scope import build_scope, ListScope, BaseScope
from ..common.exceptions import RepairException
from ..common import recursive_apply_transform, print_info_log_rank_0


def build_repair(config):
    if config.repair_type is None:
        return None
    elif config.repair_type == RepairAPI.ToCPU:
        return RepairAPI_toCPU(config)
    elif config.repair_type == RepairAPI.RaisePrecision:
        return RepairAPI_raise(config)
    else:
        raise RepairException(RepairException.InvalidRepairType, f"精度修复类型"
            f"须配置为'{RepairAPI.ToCPU}'或'{RepairAPI.RaisePrecision}，"
            f"实际配置为{config.repair_type}")


class RepairAPI(ABC):
    ToCPU = "cpu"
    RaisePrecision = "raise"

    def __init__(self, config):
        self.config = config
        self.scope = build_scope(ListScope, config.repair_scope, config.repair_api_str)
        self.saved, self.towards = "None",  "None"

    def check_name_and_module_type(self, name, module_type):
        if module_type == BaseScope.Module_Type_Module:
            return False
        if not self.scope.check(name):
            return False
        return True

    def convert(self, name, module_type, args, kwargs):
        is_target = self.check_name_and_module_type(name, module_type)
        if is_target:
            args = recursive_apply_transform(args, self.fx)
            kwargs = recursive_apply_transform(kwargs, self.fx)
            print_info_log_rank_0(f"[calibrator] convert inputs of {name} to "
                                  f"{self.towards}.")
        return args, kwargs

    def invert(self, name, module_type, out_feat):
        is_target = self.check_name_and_module_type(name, module_type)
        if is_target:
            out_feat = recursive_apply_transform(out_feat, self.inv_fx)
            print_info_log_rank_0(f"[calibrator] convert outputs of {name} back to "\
                                  f"{self.saved}.")
        return out_feat


class RepairAPI_toCPU(RepairAPI):
    def fx(self, arg, _):
        if isinstance(arg, torch.Tensor):
            self.saved = arg.device
            self.towards = torch.device("cpu")
            return arg.cpu()
        return arg

    def inv_fx(self, arg, _):
        if isinstance(arg, torch.Tensor):
            return arg.to(self.saved)
        return arg


class RepairAPI_raise(RepairAPI):
    raise_dtype_map = {
        torch.bfloat16: torch.float32,
        torch.float16: torch.float32
    }

    def fx(self, arg, _):
        if isinstance(arg, torch.Tensor):
            self.saved = arg.dtype
            self.towards = RepairAPI_raise.raise_dtype_map.get(self.saved)
            # bug: nested input may be of various dtypes. which to save and invert?
            return arg.to(self.towards)
        return arg

    def inv_fx(self, arg, _):
        if isinstance(arg, torch.Tensor):
            return arg.to(self.saved)
        return arg


