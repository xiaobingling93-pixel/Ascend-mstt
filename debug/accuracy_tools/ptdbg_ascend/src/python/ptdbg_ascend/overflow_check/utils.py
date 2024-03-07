import os
import torch

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

from ..common.utils import Const, check_switch_valid, check_inplace_op, OverflowConst
from ..dump.dump import dump_stack_info, get_scalar_data_info, dump_data_by_rank_count, \
    get_not_float_tensor_info, get_float_tensor_info
from ..dump.utils import DumpUtil, make_dump_data_dir


class OverFlowUtil(object):
    overflow_check_switch = None
    overflow_filter_switch = Const.OFF
    real_overflow_dump_times = 0
    overflow_nums = 1

    @staticmethod
    def set_overflow_check_switch(switch, filter_switch):
        OverFlowUtil.overflow_check_switch = switch
        OverFlowUtil.overflow_filter_switch = filter_switch

    @staticmethod
    def get_overflow_check_switch():
        if OverFlowUtil.overflow_check_switch is None:
            return True
        return OverFlowUtil.overflow_check_switch == "ON"

    @staticmethod
    def inc_overflow_dump_times():
        OverFlowUtil.real_overflow_dump_times += 1

    @staticmethod
    def check_overflow_dump_times(need_dump_times):
        if need_dump_times == -1:
            return True
        return OverFlowUtil.real_overflow_dump_times < need_dump_times


def set_overflow_check_switch(switch, filter_switch=Const.OFF):
    check_switch_valid(switch)
    check_switch_valid(filter_switch)

    OverFlowUtil.set_overflow_check_switch(switch, filter_switch)


def dump_overflow(module_name, in_feat, out_feat, dump_file):
    name_template = f"{module_name}" + "_{}"
    DumpUtil.dump_data_dir = make_dump_data_dir(dump_file)
    dump_stack_info(name_template)
    if check_inplace_op(name_template):
        if Const.PRE_FORWARD in name_template:
            name_template = name_template.replace(Const.PRE_FORWARD, Const.FORWARD)
        else:
            _dump_tensor_completely(in_feat, name_template.format("output"))
            return

    if "forward" in name_template:
        _dump_tensor_completely(in_feat, name_template.format("input"))
        _dump_tensor_completely(out_feat, name_template.format("output"))
    else:
        _dump_tensor_completely(in_feat, name_template.format("output"))
        _dump_tensor_completely(out_feat, name_template.format("input"))


def _dump_tensor_completely(x, prefix):
    dump_flag = Const.DUMP_RATIO_MAX + 1
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            _dump_tensor_completely(item, "{}.{}".format(prefix, i))
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0 or len(x.shape) == 0 or not x.is_floating_point():
            if OverFlowUtil.overflow_filter_switch == Const.OFF:
                data_info = get_not_float_tensor_info(x)
                dump_data_by_rank_count(dump_flag, prefix, data_info)
        else:
            data_info = get_float_tensor_info(x)
            dump_data_by_rank_count(dump_flag, prefix, data_info)

    elif OverFlowUtil.overflow_filter_switch == Const.OFF:
        if isinstance(x, bool) or isinstance(x, int) or isinstance(x, float):
            data_info = get_scalar_data_info(x)
            dump_data_by_rank_count(dump_flag, prefix, data_info)


def overflow_debug_mode_enalbe():
    overflow_mode = os.getenv(OverflowConst.OVERFLOW_DEBUG_MODE_ENABLE, Const.ENV_DISABLE)
    return overflow_mode == Const.ENV_ENABLE


def check_overflow_npu():
    if overflow_debug_mode_enalbe():
        float_status = torch.zeros(8).npu()
        result = torch_npu.npu_get_float_status(float_status, OverflowConst.OVERFLOW_DEBUG_MODE)
        if (result.cpu()[0] != 0):
            return True
        else:
            return False
    else:
        return torch_npu._C._check_overflow_npu()


def clear_overflow_npu():
    if overflow_debug_mode_enalbe():
        float_status = torch.zeros(8).npu()
        torch_npu.npu_clear_float_status(float_status, OverflowConst.OVERFLOW_DEBUG_MODE)
    else:
        torch_npu._C._clear_overflow_npu()