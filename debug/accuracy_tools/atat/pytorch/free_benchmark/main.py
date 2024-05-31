import importlib
from abc import ABC

import torch
from atat.pytorch.free_benchmark import Const, print_error_log_rank_0

from atat.pytorch.free_benchmark.common.params import data_pre_deal, make_handler_params
from atat.pytorch.free_benchmark.common.enums import (
    PerturbationMode,
    FuzzLevel,
    DeviceType,
)

class FreeBenchmarkCheck(ABC):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        if self.config.pert_mode is None:
            self.config.pert_mode = PerturbationMode.IMPROVE_PRECISION
        if self.config.fuzz_level is None:
            self.config.fuzz_level = FuzzLevel.BASE_LEVEL
        if self.config.fuzz_device is None:
            self.config.fuzz_device = DeviceType.NPU
        self.current_iter = 0

    def update_iter(self, update_iter):
        self.current_iter = update_iter

    def pre_forward(self, name, module, data_processor, args, kwargs):
        if not self.config.fuzz_stage == Const.BACKWARD:
            return
            # TODO 只支持check模式
        origin_func = (
            module._slow_forward if torch._C._get_tracing_state() else module.forward
        )

    def forward(self, name, module, args, kwargs, output):
        if not self.config.fuzz_stage == Const.FORWARD:
            return output, []
        origin_func = (
            module._slow_forward if torch._C._get_tracing_state() else module.forward
        )
        data_params = data_pre_deal(name, origin_func, args, kwargs)
        if data_params.index == -1:
            return output, []
        data_params.original_result = output
        data_params.fuzz_stage = self.config.fuzz_stage


    def backward(self, name, module, grad_output):

        if not self.config.fuzz_stage == Const.BACKWARD:
            return
        try:
            grad_saver = getattr(module, "grad_saver")
        except AttributeError:
            print_error_log_rank_0(
                f"[atat] Free benchmark:  get grad saver failed. api_name:{name}"
            )
            return

        _new_grad_output = grad_output
        try:
            need_grad_tensors, _inner_args = grad_saver.get_vjp_input()
            origin_grad_input = grad_saver.get_grad_input_from_vjp(
                tuple(need_grad_tensors), _new_grad_output, _inner_args
            )
            grad_saver.origin_grad_input = tuple([x.cpu() for x in origin_grad_input])
            grad_saver.calculate_perturbed_grad_input(
                _new_grad_output, need_grad_tensors, _inner_args
            )
        except Exception as e:
            print_error_log_rank_0(
                f"[atat] Free benchmark: grad vjp calculate failed. api_name:{name} error: {e}"
            )
            return
