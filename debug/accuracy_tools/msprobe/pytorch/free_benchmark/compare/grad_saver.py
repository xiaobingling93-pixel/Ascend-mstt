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

import torch
from msprobe.core.common.exceptions import FreeBenchmarkException
from msprobe.pytorch.free_benchmark import logger
from msprobe.pytorch.free_benchmark.common.constant import CommonField
from msprobe.pytorch.free_benchmark.common.params import (
    DataParams,
    HandlerParams,
    data_pre_deal,
)
from msprobe.pytorch.free_benchmark.perturbed_layers.layer_factory import LayerFactory
from msprobe.pytorch.free_benchmark.result_handlers.handler_factory import (
    FuzzHandlerFactory,
)


class GradSaver:

    def __init__(self, origin_func, handler_params: HandlerParams):

        self.handler_params = handler_params
        self.api_name = handler_params.api_name
        self.origin_func = origin_func
        self.is_compare = True
        self.kwargs = dict()
        self.perturbed_grad_input = tuple()
        self.origin_grad_input = tuple()
        self.need_grad_flag = list()
        self.backward_input = tuple()

    def register_compare_func_for_inputs(self, inputs, data_processor):
        _index = 0
        for j, obj in enumerate(inputs):
            if torch.is_tensor(obj) and obj.requires_grad:

                def compare_func(grad, new_grad_index=_index, input_index=j):
                    if not self.is_compare:
                        return grad
                    try:
                        perturbed_grad = self.check_grad_input(grad, new_grad_index)
                        handler = FuzzHandlerFactory.create(self.handler_params)
                        self.compare_grad_results(
                            handler, grad, perturbed_grad, index=input_index
                        )
                        data_processor.update_unequal_rows(handler.get_unequal_rows())
                    except IndexError:
                        logger.warning_on_rank_0(
                            f"[msprobe] Free benchmark: grad index out of range. api:{self.handler_params.api_name}."
                            f"index:{new_grad_index}, perturbation grad len {len(self.perturbed_grad_input)}"
                        )
                        return grad
                    except FreeBenchmarkException as e:
                        logger.warning_on_rank_0(
                            f"[msprobe] Free benchmark: grad input check error: {e}"
                        )
                        return grad
                    except Exception as e:
                        logger.warning_on_rank_0(
                            f"[msprobe] Free benchmark: grad compare error: {e}"
                        )
                        return grad
                    return grad

                obj.register_hook(compare_func)
                _index += 1

    def compare_grad_results(self, handler, origin_grad, perturbed_grad, index):
        data_params = DataParams()
        data_params.original_result = origin_grad
        data_params.perturbed_result = perturbed_grad
        data_params.grad_unequal_flag = False
        data_params.valid_input_index = index
        try:
            handler.handle(data_params)
            if not data_params.is_consistent:
                self.is_compare = False
                data_params.grad_unequal_flag = True
                data_params.is_consistent = True
                data_params.perturbed_result = self.perturbed_grad_input
                data_params.original_result = self.origin_grad_input
                handler.handle(data_params)
        except Exception as e:
            logger.warning_on_rank_0(
                f"[msprobe] Free benchmark: compare two vjp failed: api:{self.handler_params.api_name}."
                f"{e}"
            )

    def check_grad_input(self, origin_grad, new_grad_index):
        if self.perturbed_grad_input is None:
            raise FreeBenchmarkException(
                FreeBenchmarkException.InvalidGrad,
                f"grad not exists : {self.api_name}.",
            )
        with torch.no_grad():
            perturbed_grad = self.perturbed_grad_input[new_grad_index].to(
                origin_grad.device
            )
        if origin_grad.shape != perturbed_grad.shape:
            raise FreeBenchmarkException(
                FreeBenchmarkException.InvalidGrad,
                f"grad shapes are inconsistent. api:{self.handler_params.api_name}."
                f"origin:{origin_grad.shape}, perturbation: {perturbed_grad.shape}",
            )
        return perturbed_grad

    def cache_backward_input(self, backward_input_list):
        _inputs = []
        with torch.no_grad():
            for backward_input in backward_input_list:
                if torch.is_tensor(backward_input):
                    _inputs.append(
                        {
                            CommonField.DEVICE: backward_input.device,
                            CommonField.FUZZ_TENSOR: backward_input.cpu(),
                            CommonField.REQUIRES_GRAD: backward_input.requires_grad,
                        }
                    )
                else:
                    _inputs.append(backward_input)
        self.backward_input = _inputs

    def get_vjp_input(self):
        inner_args_tmp = []
        need_grad_tensors = []
        for object_ in self.backward_input:
            if isinstance(object_, dict) and CommonField.FUZZ_TENSOR in object_.keys():
                tensor_ = torch.tensor(
                    object_.get(CommonField.FUZZ_TENSOR).data,
                    dtype=object_.get(CommonField.FUZZ_TENSOR).dtype,
                    device=object_.get(CommonField.DEVICE),
                    requires_grad=object_.get(CommonField.REQUIRES_GRAD),
                )

                if tensor_.requires_grad:
                    inner_args_tmp.append(CommonField.HOLD_PLACE)
                    need_grad_tensors.append(tensor_)
                    self.need_grad_flag.append(True)
                else:
                    self.need_grad_flag.append(False)
                    inner_args_tmp.append(tensor_)
            else:
                self.need_grad_flag.append(False)
                inner_args_tmp.append(object_)

        return need_grad_tensors, tuple(inner_args_tmp)

    def get_grad_input_from_vjp(self, need_grad_tensors, grad_output, inner_args):
        def vjp_func(*inputs):
            _real_input = []
            index_ = 0
            for object_ in inner_args:
                if object_ is CommonField.HOLD_PLACE:
                    _real_input.append(inputs[index_])
                    index_ += 1
                else:
                    _real_input.append(object_)
            kwargs = self.kwargs.copy()
            if "inplace" in kwargs:
                kwargs["inplace"] = False
            return self.origin_func(*_real_input, **kwargs)

        _, grad_input = torch.autograd.functional.vjp(
            vjp_func, tuple(need_grad_tensors), grad_output
        )
        return grad_input

    def calculate_perturbed_grad_input(
        self, grad_output, need_grad_tensors, inner_args
    ):
        data_params = data_pre_deal(
            self.handler_params.api_name,
            self.get_grad_input_from_vjp,
            [need_grad_tensors, grad_output, inner_args],
            {},
        )
        layer = LayerFactory.create(
            self.handler_params.api_name,
            self.handler_params.fuzz_device,
            self.handler_params.pert_mode,
        )
        layer.handle(data_params)
        # 确定扰动成功后，才会暂存
        if data_params.perturbed_result:
            self.perturbed_grad_input = tuple(
                [x.cpu() for x in data_params.perturbed_result]
            )
