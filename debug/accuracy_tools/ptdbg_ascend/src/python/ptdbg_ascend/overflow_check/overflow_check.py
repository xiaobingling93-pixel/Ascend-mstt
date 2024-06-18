import os
from pathlib import Path

import numpy as np
import torch

try:
    import torch_npu
except ImportError:
    is_gpu = True
else:
    is_gpu = False

from ..common.utils import print_warn_log, get_time, print_info_log, Const
from ..dump.dump import forward_init_status, forward_acl_dump
from .utils import OverFlowUtil, dump_overflow, check_overflow_npu, clear_overflow_npu
from ..dump.utils import DumpUtil, Const, get_tensor_rank, create_dirs_if_not_exist, check_single_rank_folder
from .info_dump import write_api_info_json, ForwardAPIInfo, BackwardAPIInfo
from ..dump import dump
from ..common.file_check_util import FileCheckConst

backward_init_status = False
api_overflow = []
forward_api_info = {}
backward_api_info = {}
FORWARD_REAL_DATA_PATH = os.path.join('./', 'forward_real_data')
BACKWARD_REAL_DATA_PATH = os.path.join('./', 'backward_real_data')
rank = os.getpid()
pkl_name = ''


def check_overflow_environment(pid):
    if not OverFlowUtil.get_overflow_check_switch():
        return False
    if pid != os.getpid():
        return False
    if is_gpu:
        print_warn_log("Overflow detection is not supported in the GPU environment.")
        return False
    global backward_init_status
    if backward_init_status or forward_init_status:
        return False
    return True


def check_data_overflow(x):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            if True == check_data_overflow(item):
                return True
        return False
    else:
        if isinstance(x, torch.Tensor) and x.numel() != 0 and x.dtype != torch.bool:
            if x.is_meta:
                return False
            if len(x.shape) == 0:
                tensor_max = x.cpu().detach().float().numpy().tolist()
                tensor_min = tensor_max
            elif torch.is_complex(x):
                data_np = x.detach().cpu().numpy()
                data_abs = np.abs(data_np)
                tensor_max = np.max(data_abs).item()
                tensor_min = np.min(data_abs).item()
            else:
                tensor_max = torch._C._VariableFunctionsClass.max(x).cpu().detach().float().numpy().tolist()
                tensor_min = torch._C._VariableFunctionsClass.min(x).cpu().detach().float().numpy().tolist()
            # inf
            if tensor_max == float('inf') or tensor_min == float('-inf'):
                return True
            if x.dtype in [torch.float16, torch.float32, torch.bfloat16] and \
                    (tensor_max == torch.finfo(x.dtype).max or tensor_min == torch.finfo(x.dtype).min):
                return True
            # nan
            elif tensor_max != tensor_max or tensor_min != tensor_min:
                return True
            else:
                return False
        elif isinstance(x, bool) or isinstance(x, int) or isinstance(x, float):
            if x == float('inf') or x == float('-inf') or x != x:
                return True
            else:
                return False
        else:
            return False


def check_path(apis, path):
    return any(api in path for api in apis)


def overflow_check(name, **kwargs):
    overflow_nums = OverFlowUtil.overflow_nums
    pid = kwargs.get('pid')
    dump_mode = DumpUtil.dump_switch_mode
    if not pid:
        return RuntimeError("Not get the specified process pid.")

    def overflowcheck_hook(module, in_feat, out_feat=None):
        if not check_overflow_environment(pid):
            return
        dump_file = DumpUtil.get_dump_path()
        global rank
        dump_dir, dump_filename = os.path.split(dump_file)
        dump_dir = os.path.join(dump_dir, "step{}".format(DumpUtil.iter_num))
        if not os.path.exists(dump_dir):
            Path(dump_dir).mkdir(mode=FileCheckConst.DATA_DIR_AUTHORITY, exist_ok=True)
        if DumpUtil.is_single_rank is None:
            DumpUtil.is_single_rank = check_single_rank_folder(dump_dir)
        dump_file = os.path.join(dump_dir, dump_filename)
        rank_this = get_tensor_rank(in_feat, out_feat)
        DumpUtil.dump_root = os.path.dirname(DumpUtil.dump_path)
        if rank_this is not None and rank != rank_this:
            rank = rank_this
            dump.rename_()
        if DumpUtil.target_rank is not None:
            if rank != DumpUtil.target_rank:
                return
        dump_path = create_dirs_if_not_exist(rank, dump_file)
        global pkl_name
        pkl_name = dump_path
        dump_dir = os.path.split(dump_path)[0]
        global api_overflow
        global forward_api_info
        global backward_api_info

        module_name = name
        if hasattr(torch_npu._C, '_npu_is_support_inf_nan') and torch_npu._C._npu_is_support_inf_nan():
            # backward API endwith backward
            if module_name.endswith(Const.BACKWARD):
                check_feat = in_feat
            else:
                check_feat = out_feat
            module.has_overflow = check_data_overflow(check_feat)
        else:
            module.has_overflow = check_overflow_npu()
        if not module.has_overflow:
            if hasattr(module, 'input_args'):
                del module.input_args
            if hasattr(module, 'input_kwargs'):
                del module.input_kwargs
        if module.has_overflow and OverFlowUtil.check_overflow_dump_times(overflow_nums):
            if overflow_type_judge(in_feat, out_feat, module_name) and DumpUtil.need_replicate:
                if module_name.endswith(Const.FORWARD):
                    forward_api_info.update({name: ForwardAPIInfo(name, True, module.input_args, module.input_kwargs)})
                    api_overflow.append(module_name)
                else:
                    api_overflow.append(module_name.replace("backward", "forward"))
                    backward_api_info.update({name: BackwardAPIInfo(name, out_feat)})
            OverFlowUtil.inc_overflow_dump_times()
            dump_file_name = os.path.join(dump_dir,
                                          "{}_{}.pkl".format(module_name.replace(Const.DELIMITER, '_'), OverFlowUtil.real_overflow_dump_times))
            dump_overflow(module_name, in_feat, out_feat, dump_file_name)
            dump.pkl_name = dump_file_name

            print_warn_log("[overflow {} times]: module name :'{}' is overflow and dump file is saved in '{}'."
                           .format(OverFlowUtil.real_overflow_dump_times, module_name,
                                   os.path.realpath(dump_file_name)))
            if dump_mode == "acl":
                acl_dump(module, module_name)
            dump.write_to_disk()
            # clear overflow flag for the next check
            clear_overflow_npu()
            if not OverFlowUtil.check_overflow_dump_times(overflow_nums):
                for key in forward_api_info:
                    write_api_info_json(forward_api_info[key])
                for key in backward_api_info:
                    write_api_info_json(backward_api_info[key])
                raise ValueError("[overflow {} times]: dump file is saved in '{}'."
                                 .format(OverFlowUtil.real_overflow_dump_times, os.path.realpath(dump_file_name)))

    def overflow_type_judge(in_feat, out_feat, module_name):
        if module_name.endswith(Const.BACKWARD):
            check_feat = out_feat
        else:
            check_feat = in_feat
        if check_data_overflow(check_feat):
            print_warn_log("module name :'{}' is overflow and its inputs already has an overflow, so you need "
                           "to go back to find where the overflow started.".format(module_name))
            return False
        elif not check_data_overflow(in_feat) and not check_data_overflow(out_feat):
            print_warn_log("module name :'{}' is overflow and its inputs and outputs do not overflow, "
                           "so this is a process overflow".format(module_name))
            return False
        else:
            print_warn_log("module name :'{}' is overflow. Its input is normal and its output "
                           "is overflow.".format(module_name))
            return True

    def acl_dump(module, module_name):
        if "forward" in module_name:
            forward_acl_dump(module, module_name)
        if "backward" in module_name:
            print_info_log("The overflow is caused by backward operator {}. "
                           "You can use reverse acl dump(mode='acl') to get operator dump data.".format(module_name))

    return overflowcheck_hook
