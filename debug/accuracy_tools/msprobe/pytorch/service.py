import functools
import os
import time
from pathlib import Path

from msprobe.core.common.const import Const, FileCheckConst
from msprobe.core.common.exceptions import DistributedNotInitializedError, MsprobeException
from msprobe.core.common.file_check import FileChecker, check_path_before_create
from msprobe.core.data_dump.data_collector import build_data_collector
from msprobe.core.data_dump.data_processor.base import ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs
from msprobe.core.data_dump.scope import BaseScope
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import get_rank_if_initialized
from msprobe.pytorch.hook_module import remove_dropout
from msprobe.pytorch.hook_module.api_registry import api_register


class Service:
    def __init__(self, config):
        self.model = None
        self.config = config
        self.data_collector = build_data_collector(config)
        self.module_processor = ModuleProcesser(self.data_collector.scope)
        self.switch = False
        self.current_iter = 0
        self.first_start = True
        try:
            self.current_rank = get_rank_if_initialized()
        except DistributedNotInitializedError:
            self.current_rank = None
        self.dump_iter_dir = None
        if self.config.online_run_ut:
            attl_config = ATTLConfig(is_benchmark_device=False,
                                     connect_ip=self.config.host,
                                     connect_port=self.config.port,
                                     nfs_path=self.config.nfs_path)
            need_dump = len(self.config.rank) == 0 or self.current_rank in self.config.rank
            self.attl = ATTL('npu', attl_config, need_dump=need_dump)
            if self.config.nfs_path:
                self.attl.upload("start")

    @staticmethod
    def forward_backward_dump_end():
        logger.info_on_rank_0("Data needed ends here.")
        api_register.api_originality()

    def build_hook(self, module_type, name):
        def pre_hook(api_or_module_name, module, args, kwargs):
            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name
            self.data_collector.visit_and_clear_overflow_status(api_or_module_name)

            if not self.switch:
                return args, kwargs
            if self.data_collector:
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)
                self.data_collector.pre_forward_data_collect(api_or_module_name, module, pid, module_input_output)
            return args, kwargs

        def forward_hook(api_or_module_name, module, args, kwargs, output):
            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name
            self.data_collector.visit_and_clear_overflow_status(api_or_module_name)

            if not self.switch:
                return None

            if self.config.online_run_ut:
                if not self.data_collector.scope or self.data_collector.scope.check(api_or_module_name):
                    return None
                api_data = ApiData(name[:-1], args, kwargs, output, self.current_iter, self.current_rank)
                self.attl_send(api_data)
                return None

            if self.data_collector:
                module_input_output = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=output)
                self.data_collector.forward_data_collect(api_or_module_name, module, pid, module_input_output)
                if self.data_collector.if_return_forward_new_output():
                    return self.data_collector.get_forward_new_output()
            return output

        def backward_hook(api_or_module_name, module, grad_input, grad_output):
            if module_type == BaseScope.Module_Type_Module:
                api_or_module_name = module.mindstudio_reserved_name
            self.data_collector.visit_and_clear_overflow_status(api_or_module_name)

            if not self.switch:
                return

            if self.config.online_run_ut:
                if not self.data_collector.scope or self.data_collector.scope.check(api_or_module_name):
                    return None
                api_data = ApiData(name[:-1], grad_input, {}, grad_output, self.current_iter, self.current_rank)
                self.attl_send(api_data)
                return None

            if self.data_collector:
                # 此处获取到的grad_input实际为反向过程的输出数据，grad_output为反向过程的输入数据，因此传入时调换顺序
                module_input_output = ModuleBackwardInputsOutputs(grad_input=grad_output, grad_output=grad_input)
                self.data_collector.backward_data_collect(api_or_module_name, module, pid, module_input_output)

        pid = os.getpid()
        forward_name_template = name + Const.FORWARD
        backward_name_template = name + Const.BACKWARD
        pre_forward_hook = functools.partial(pre_hook, forward_name_template)
        forward_hook = functools.partial(forward_hook, forward_name_template)
        backward_hook = functools.partial(backward_hook, backward_name_template)
        return pre_forward_hook, forward_hook, backward_hook

    def step(self):
        self.current_iter += 1
        self.data_collector.update_iter(self.current_iter)

        ModuleProcesser.reset_module_stats()
        HOOKModule.reset_module_stats()

    def start(self, model, api_origin=False):
        self.model = model
        if self.config.step and self.current_iter > max(self.config.step):
            # send end or step signal
            if self.config.online_run_ut:
                if self.config.nfs_path:
                    self.attl.upload("end")
                elif self.attl.socket_manager is not None:
                    logger.debug(f"进程{os.getpid()} 已完成,准备发送STOP信号")
                    self.attl.socket_manager.send_stop_signal()
                else:
                    # current rank not need dump, wait
                    while True:
                        time.sleep(2)
            self.stop()
            raise Exception("msprobe: exit after iteration {}".format(max(self.config.step)))
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.first_start:
            if self.config.rank and self.current_rank not in self.config.rank:
                return
            self.register_hook_new()
            self.first_start = False
        if api_origin:
            api_register.api_modularity()
        self.switch = True
        logger.info_on_rank_0(f"Dump switch is turned on at step {self.current_iter}. ")
        if self.config.level != "L2":
            self.create_dirs()
            logger.info_on_rank_0(f"Dump data will be saved in {self.dump_iter_dir}.")

    def stop(self):
        if self.config.level == "L2":
            return
        if self.config.step and self.current_iter not in self.config.step:
            return
        if self.config.rank and self.current_rank not in self.config.rank:
            return
        self.switch = False
        self.data_collector.write_json()

    def create_dirs(self):
        check_path_before_create(self.config.dump_path)
        if not os.path.exists(self.config.dump_path):
            Path(self.config.dump_path).mkdir(mode=0o750, exist_ok=True)
        file_check = FileChecker(self.config.dump_path, FileCheckConst.DIR)
        file_check.common_check()
        self.dump_iter_dir = os.path.join(self.config.dump_path, f"step{self.current_iter}")
        cur_rank = self.current_rank if self.current_rank is not None else ''
        dump_dir = os.path.join(self.dump_iter_dir, f"rank{cur_rank}")
        if not os.path.exists(dump_dir):
            Path(dump_dir).mkdir(mode=0o750, parents=True, exist_ok=True)
        if self.config.task in self.data_collector.tasks_need_tensor_data:
            dump_data_dir = os.path.join(dump_dir, "dump_tensor_data")
            Path(dump_data_dir).mkdir(mode=0o750, exist_ok=True)
        else:
            dump_data_dir = None

        dump_file_path = os.path.join(dump_dir, "dump.json")
        stack_file_path = os.path.join(dump_dir, "stack.json")
        construct_file_path = os.path.join(dump_dir, "construct.json")
        free_benchmark_file_path = os.path.join(self.config.dump_path, "free_benchmark.csv")
        self.data_collector.update_dump_paths(
            dump_file_path, stack_file_path, construct_file_path, dump_data_dir, free_benchmark_file_path)

    def register_hook_new(self):
        logger.info_on_rank_0("The {} hook function is successfully mounted to the model.".format(self.config.task))
        if self.config.level in ["L0", "mix"]:
            if self.model is None:
                logger.error_log_with_exp("The model is None.", MsprobeException.INVALID_PARAM_ERROR)
            logger.info_on_rank_0("The init dump mode is enabled, and the module dump function will not be available")
            for name, module in self.model.named_modules():
                if module == self.model:
                    continue
                prefix = BaseScope.Module_Type_Module + Const.SEP + name + Const.SEP + \
                         module.__class__.__name__ + Const.SEP

                pre_forward_hook, forward_hook, backward_hook = self.build_hook(BaseScope.Module_Type_Module, prefix)
                module.register_forward_hook(forward_hook, with_kwargs=True)
                module.register_full_backward_hook(backward_hook)

                module.register_forward_pre_hook(
                    self.module_processor.node_hook(prefix + Const.FORWARD, Const.START))
                module.register_forward_hook(
                    self.module_processor.node_hook(prefix + Const.FORWARD, Const.STOP))
                module.register_full_backward_pre_hook(
                    self.module_processor.node_hook(prefix + Const.BACKWARD, Const.START))
                module.register_full_backward_hook(
                    self.module_processor.node_hook(prefix + Const.BACKWARD, Const.STOP))

        if self.config.level in ["mix", "L1", "L2"]:
            api_register.initialize_hook(functools.partial(self.build_hook, BaseScope.Module_Type_API))
            api_register.api_modularity()

        if Const.STATISTICS == self.config.task or Const.TENSOR == self.config.task:
            remove_dropout()

    def attl_send(self, api_data):
        logger.info(f"tools is dumping api: {api_data.name}, rank: {self.current_rank}")
        if self.config.nfs_path:
            self.attl.upload(api_data)
        else:
            self.attl.send(api_data)
