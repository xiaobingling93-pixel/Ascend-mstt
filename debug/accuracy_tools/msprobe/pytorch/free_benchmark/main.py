# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC

import torch
from msprobe.core.common.const import Const
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.constant import CommonField
from msprobe.pytorch.free_benchmark.common.enums import (
    DeviceType,
    FuzzLevel,
    HandlerType,
    PerturbationMode,
)
from msprobe.pytorch.free_benchmark.common.params import (
    data_pre_deal,
    make_handler_params,
)
from msprobe.pytorch.free_benchmark.compare.grad_saver import GradSaver
from msprobe.pytorch.free_benchmark.perturbed_layers.layer_factory import LayerFactory
from msprobe.pytorch.free_benchmark.result_handlers.handler_factory import (
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
        if self.config.handler_type == HandlerType.FIX:
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
        setattr(module, CommonField.GRADSAVER, grad_saver)

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
        perturbed_output = handler.handle(data_params)
        return perturbed_output, handler.get_unequal_rows()

    def backward(self, name, module, grad_output):

        if not self.config.fuzz_stage == Const.BACKWARD:
            return
        try:
            grad_saver = getattr(module, CommonField.GRADSAVER)
        except AttributeError:
            logger.warning_on_rank_0(
                f"[msprobe] Free benchmark:  get grad saver failed. api_name:{name}"
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
                f"[msprobe] Free benchmark: grad vjp calculate failed. api_name:{name} error: {e}"
            )
            return
