import os
import torch
from ..common.utils import Const, check_switch_valid, generate_compare_script, check_is_npu, print_error_log, \
    CompareException, print_warn_log
from ..dump.dump import DumpUtil, acc_cmp_dump, write_to_disk, get_pkl_file_path, reset_module_count, GLOBAL_THREAD_POOL
from ..dump.utils import set_dump_path, set_dump_switch_print_info, generate_dump_path_str, \
        set_dump_switch_config, set_backward_input
from ..overflow_check.utils import OverFlowUtil
from ..overflow_check.overflow_check import overflow_check
from ..hook_module.register_hook import register_hook_core, init_overflow_nums
from ..hook_module.hook_module import HOOKModule
from .debugger_config import DebuggerConfig


class PrecisionDebugger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PrecisionDebugger, cls).__new__(cls)
            cls._instance.first_start = True
            cls._instance.hook_func = None
            cls._instance.config = None
            cls._instance.model = None
            cls._instance.enable_dataloader = False
        return cls._instance

    def __init__(self, dump_path=None, hook_name=None, rank=None, step=None, enable_dataloader=False, model=None):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            if hook_name is None:
                err_msg = "You must provide hook_name argument to PrecisionDebugger\
                                when config is not provided."
                raise Exception(err_msg)
            self.config = DebuggerConfig(dump_path, hook_name, rank, step)
            self.configure_hook = self.get_configure_hook(self.config.hook_name)
            self.configure_hook()
            DumpUtil.target_iter = self.config.step
            DumpUtil.target_rank = self.config.rank
            set_dump_path(self.config.dump_path)
            self.hook_func = overflow_check if self.config.hook_name == "overflow_check" else acc_cmp_dump
            self.model = model
            self.enable_dataloader = enable_dataloader
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
            input_output_mode=[Const.ALL], acl_config=None, backward_input=None, summary_only=False, summary_mode=None):
        if mode == "acl" and self.model is not None:
            print_error_log("Init dump does not support ACL dump mode.")
            raise CompareException(CompareException.INVALID_DUMP_MODE)
        scope = scope if scope is not None else []
        api_list = api_list if api_list is not None else []
        backward_input = backward_input if backward_input is not None else []

        if summary_only:
            if summary_mode is not None:
                raise ValueError("summary_mode can not be used with summary_only")
            print_warn_log("Argument 'summary_only' will be deprecated, it would be better to use 'summary_mode'")
            summary_mode = "summary"
        elif summary_mode is None:
            summary_mode = "all"

        set_dump_switch_config(mode=mode, scope=scope, api_list=api_list,
                               filter_switch=filter_switch, dump_mode=input_output_mode, summary_only=summary_only,
                               summary_mode=summary_mode)
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
        instance = cls._instance
        if not instance:
            raise Exception("No instance of PrecisionDebugger found.")
        if instance.enable_dataloader:
            print_warn_log("DataLoader is enabled, start() skipped.")
        else:
            if DumpUtil.iter_num in DumpUtil.target_iter or not DumpUtil.target_iter:
                if instance.first_start:
                    register_hook_core(instance.hook_func, instance.model)
                    instance.first_start = False
                DumpUtil.dump_switch = "ON"
                OverFlowUtil.overflow_check_switch = "ON"
                dump_path_str = generate_dump_path_str()
                set_dump_switch_print_info("ON", DumpUtil.dump_switch_mode, dump_path_str)
            elif DumpUtil.target_iter and DumpUtil.iter_num > max(DumpUtil.target_iter):
                cls.stop()
                raise Exception("ptdbg: exit after iteration {}".format(max(DumpUtil.target_iter)))
            else:
                cls.stop()

    @classmethod
    def stop(cls):
        instance = cls._instance
        if not instance:
            raise Exception("PrecisionDebugger instance is not created.")
        if instance.enable_dataloader:
            print_warn_log("DataLoader is enabled, stop() skipped.")
        else:
            DumpUtil.dump_switch = "OFF"
            OverFlowUtil.overflow_check_switch = "OFF"
            dump_path_str = generate_dump_path_str()
            set_dump_switch_print_info("OFF", DumpUtil.dump_switch_mode, dump_path_str)
            write_to_disk()
            if DumpUtil.is_single_rank:
                GLOBAL_THREAD_POOL.shutdown(wait=True)
            if check_is_npu() and DumpUtil.dump_switch_mode in [Const.ALL, Const.API_STACK, Const.LIST, Const.RANGE, Const.API_LIST]:
                generate_compare_script(DumpUtil.dump_data_dir, get_pkl_file_path(), DumpUtil.dump_switch_mode)

    @classmethod
    def step(cls):
        instance = cls._instance
        if not instance:
            raise Exception("PrecisionDebugger instance is not created.")
        if not instance.enable_dataloader:
            DumpUtil.iter_num += 1
            DumpUtil.dump_init_enable = True
            HOOKModule.module_count = {}
            reset_module_count()
        else:
            print_warn_log("DataLoader is enabled, step() skipped.")

    @staticmethod
    def incr_iter_num_maybe_exit():
        PrecisionDebugger.step()
        PrecisionDebugger.start()


def iter_tracer(func):
    def func_wrapper(*args, **kwargs):
        debugger_instance = PrecisionDebugger._instance
        temp_enable_dataloader = debugger_instance.enable_dataloader
        debugger_instance.enable_dataloader = False
        debugger_instance.stop()
        result = func(*args, **kwargs)
        debugger_instance.incr_iter_num_maybe_exit()
        debugger_instance.enable_dataloader = temp_enable_dataloader
        return result
    return func_wrapper
