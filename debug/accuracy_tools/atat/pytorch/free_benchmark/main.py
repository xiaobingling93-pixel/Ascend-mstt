from abc import ABC

import torch
from atat.pytorch.free_benchmark import Const, logger
from atat.pytorch.free_benchmark.common.params import data_pre_deal, make_handler_params
from atat.pytorch.free_benchmark.common.enums import (
    PerturbationMode,
    FuzzLevel,
    DeviceType,
    HandlerType
)
from atat.pytorch.free_benchmark.compare.grad_saver import GradSaver
from atat.pytorch.free_benchmark.perturbed_layers.layer_factory import LayerFactory
from atat.pytorch.free_benchmark.result_handlers.handler_factory import (
    FuzzHandlerFactory,
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
    
    def if_fix(self):
        if self.config.handler_type==HandlerType.FIX:
            return True
        return False

    def pre_forward(self, name, module, data_processor, args, kwargs):
        if not self.config.fuzz_stage == Const.BACKWARD:
            return
        origin_func = (
            module._slow_forward if torch._C._get_tracing_state() else module.forward
        )
        handler_params = make_handler_params(name, self.config, self.current_iter)
        grad_saver = GradSaver(origin_func, handler_params)
        grad_saver.kwargs = kwargs
        grad_saver.register_compare_func_for_inputs(args, data_processor)
        grad_saver.cache_backward_input(args)
        setattr(module, "grad_saver", grad_saver)

    def forward(self, name, module, args, kwargs, output):
        if not self.config.fuzz_stage == Const.FORWARD:
            return output, []
        origin_func = (
            module._slow_forward if torch._C._get_tracing_state() else module.forward
        )
        data_params = data_pre_deal(name, origin_func, args, kwargs)
        if data_params.valid_input_index == -1:
            return output, []
        data_params.original_result = output
        data_params.fuzz_stage = self.config.fuzz_stage

        layer = LayerFactory.create(
            name, self.config.fuzz_device, self.config.pert_mode
        )
        layer.handle(data_params)
        handler_params = make_handler_params(name, self.config, self.current_iter)
        handler = FuzzHandlerFactory.create(handler_params)
        handler.handle(data_params)
        return output, handler.get_unequal_rows()

    def backward(self, name, module, grad_output):

        if not self.config.fuzz_stage == Const.BACKWARD:
            return
        try:
            grad_saver = getattr(module, "grad_saver")
        except AttributeError:
            logger.warning_on_rank_0(
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
            logger.warning_on_rank_0(
                f"[atat] Free benchmark: grad vjp calculate failed. api_name:{name} error: {e}"
            )
            return
