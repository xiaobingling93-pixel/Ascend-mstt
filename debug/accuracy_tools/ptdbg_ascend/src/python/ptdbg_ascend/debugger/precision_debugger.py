import os
import torch
from ..common.utils import Const, check_switch_valid, generate_compare_script, check_is_npu, print_error_log, \
    CompareException
from ..dump.dump import DumpUtil, acc_cmp_dump, write_to_disk, get_pkl_file_path
from ..dump.utils import set_dump_path, set_dump_switch_print_info, generate_dump_path_str, \
        set_dump_switch_config, set_backward_input
from ..overflow_check.utils import OverFlowUtil
from ..overflow_check.overflow_check import overflow_check
from ..hook_module.register_hook import register_hook_core, init_overflow_nums
from ..hook_module.hook_module import HOOKModule
from .debugger_config import DebuggerConfig


class PrecisionDebugger:
    first_start = True
    hook_func = None
    config = None
    model = None

    def __init__(self, dump_path=None, hook_name=None, rank=None, step=None, enable_dataloader=False, model=None):
        if hook_name is None:
            err_msg = "You must provide hook_name argument to PrecisionDebugger\
                            when config is not provided."
            raise Exception(err_msg)
        step = step or []
        self.config = DebuggerConfig(dump_path, hook_name, rank, step)
        self.configure_hook = self.get_configure_hook(self.config.hook_name)
        self.configure_hook()
        DumpUtil.target_iter = self.config.step
        DumpUtil.target_rank = self.config.rank
        set_dump_path(self.config.dump_path)
        PrecisionDebugger.hook_func = overflow_check if self.config.hook_name == "overflow_check" else acc_cmp_dump
        PrecisionDebugger.model = model
        if not isinstance(enable_dataloader, bool):
            print_error_log("Params enable_dataloader only support True or False.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
        if enable_dataloader:
            DumpUtil.iter_num -= 1
            torch.utils.data.dataloader._BaseDataLoaderIter.__next__ = iter_tracer(torch.utils.data.dataloader._BaseDataLoaderIter.__next__)

    def get_configure_hook(self, hook_name):
        hook_dict = {"dump": self.configure_full_dump, "overflow_check": self.configure_overflow_dump}
        return hook_dict.get(hook_name, lambda: ValueError("hook name {} is not in ['dump', 'overflow_check']".format(hook_name)))

    def configure_full_dump(self, mode='api_stack', scope=None, api_list=None, filter_switch=Const.OFF,
            input_output_mode=[Const.ALL], acl_config=None, backward_input=None, summary_only=False):
        scope = scope or [] 
        api_list = api_list or []
        backward_input = backward_input or []
        set_dump_switch_config(mode=mode, scope=scope, api_list=api_list,
                               filter_switch=filter_switch, dump_mode=input_output_mode, summary_only=summary_only)
        if mode == 'acl':
            DumpUtil.set_acl_config(acl_config)
            if not scope or not isinstance(scope, list) or len(scope) != 1:
                raise ValueError("scope must be congfigured as a list with one api name")
            if isinstance(scope[0], str) and 'backward' in scope[0] and not backward_input:
                raise ValueError("backward_input must be configured when scope contains 'backward'")
            elif 'backward' in scope[0]:
                set_backward_input(backward_input)

    def configure_overflow_dump(self, mode="api", acl_config=None, overflow_nums=1, filter_switch=Const.OFF, need_replicate=False):
        if mode == "acl":
            DumpUtil.dump_switch_mode = mode
            DumpUtil.set_acl_config(acl_config)
        init_overflow_nums(overflow_nums)
        check_switch_valid(filter_switch)
        OverFlowUtil.overflow_filter_switch = filter_switch
        if not isinstance(need_replicate, bool):
            print_error_log("Params autojudge only support True or False.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)
        if need_replicate:
            DumpUtil.need_replicate = True

    @classmethod
    def start(cls):
        if DumpUtil.iter_num in DumpUtil.target_iter or len(DumpUtil.target_iter) == 0:
            if cls.first_start:
                register_hook_core(cls.hook_func, cls.model)
                cls.first_start = False
            DumpUtil.dump_switch = "ON"
            OverFlowUtil.overflow_check_switch = "ON"
            dump_path_str = generate_dump_path_str()
            set_dump_switch_print_info("ON", DumpUtil.dump_switch_mode, dump_path_str)
        elif len(DumpUtil.target_iter) != 0:
            if DumpUtil.iter_num > max(DumpUtil.target_iter):
                PrecisionDebugger.stop()
                raise Exception("ptdbg: exit after iteration {}".format(DumpUtil.target_iter))
        else:
            cls.stop()

    @classmethod
    def stop(cls):
        DumpUtil.dump_switch = "OFF"
        OverFlowUtil.overflow_check_switch = "OFF"
        dump_path_str = generate_dump_path_str()
        set_dump_switch_print_info("OFF", DumpUtil.dump_switch_mode, dump_path_str)
        write_to_disk()
        if check_is_npu() and DumpUtil.dump_switch_mode in [Const.ALL, Const.API_STACK, Const.LIST, Const.RANGE, Const.API_LIST]:
            generate_compare_script(DumpUtil.dump_data_dir, get_pkl_file_path(), DumpUtil.dump_switch_mode)

    @classmethod
    def step(cls):
        DumpUtil.dump_init_enable = True
        DumpUtil.iter_num += 1
        HOOKModule.module_count = {}

    @staticmethod
    def incr_iter_num_maybe_exit():
        PrecisionDebugger.step()
        PrecisionDebugger.start()


def iter_tracer(func):
    def func_wrapper(*args, **kwargs):
        PrecisionDebugger.stop()
        result = func(*args, **kwargs)
        PrecisionDebugger.incr_iter_num_maybe_exit()
        return result
    return func_wrapper