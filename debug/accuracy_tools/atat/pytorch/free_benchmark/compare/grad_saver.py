import torch
from atat.pytorch.free_benchmark import print_info_log_rank_0, print_warn_log_rank_0
from atat.pytorch.free_benchmark.common.params import DataParams, HandlerParams
from atat.pytorch.free_benchmark.common.constant import CommonField
from atat.pytorch.free_benchmark.common.utils import Tools
from atat.pytorch.free_benchmark.result_handlers.handler_factory import (
    FuzzHandlerFactory,
)
from atat.pytorch.free_benchmark.perturbed_layers.layer_factory import LayerFactory


class GradSaver:

    def __init__(self, origin_func, handler_params: HandlerParams):

        self.handler_params = handler_params
        self.api_name = handler_params.api_name
        self.origin_func = origin_func
        self.data_params = DataParams()
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
                    except Exception as e:
                        print_warn_log_rank_0(
                            f"[atat] Free benchmark: grad compara error: {e}"
                        )
                        return grad
                    return grad

                obj.register_hook(compare_func)
                _index += 1

    def compare_grad_results(self, handler, origin_grad, perturbed_grad, index):
        # TODO get dtype?
        self.data_params.original_result = origin_grad
        self.data_params.perturbed_result = perturbed_grad
        self.data_params.grad_unequal_flag = False
        self.data_params.valid_input_index = index
        try:
            handler.handle(self.data_params)
            if not self.data_params.is_consistent:
                self.is_compare = False
                self.data_params.grad_unequal_flag = True
                self.data_params.is_consistent = True
                self.data_params.perturbed_result = self.perturbed_grad_input
                self.data_params.original_result = self.origin_grad_input
                handler.handle(self.data_params)
        except Exception as e:
            print_warn_log_rank_0(
                f"[atat] Free benchmark: compare two vjp failed: api:{self.handler_params.api_name}."
                f"{e}"
            )

    def check_grad_input(self, origin_grad, new_grad_index):
        if self.perturbed_grad_input is None:
            print_info_log_rank_0(
                f"[atat] Free benchmark: grad not exsits : {self.api_name}."
            )
            return None
        try:
            with torch.no_grad():
                perturbed_grad = self.perturbed_grad_input[new_grad_index].to(
                    origin_grad.device
                )
        except IndexError:
            print_warn_log_rank_0(
                f"[atat] Free benchmark: grad index out of range. api:{self.handler_params.api_name}."
                f"index:{new_grad_index}, perturbation grad len {len(self.perturbed_grad_input)}"
            )
            return None
        if origin_grad.shape != perturbed_grad.shape:
            print_warn_log_rank_0(
                f"[atat] Free benchmark: grad shapes are unconsistent. api:{self.handler_params.api_name}."
                f"origin:{origin_grad.shape}, perturbation: {perturbed_grad.shape}"
            )
            return None
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
            if 'inplace' in kwargs:
                kwargs['inplace'] = False
            return self.origin_func(*_real_input, **kwargs)

        _, grad_input = torch.autograd.functional.vjp(
            vjp_func, tuple(need_grad_tensors), grad_output
        )
        return grad_input

    def calculate_perturbed_grad_input(self, grad_output, need_grad_tensors, inner_args):
        self.data_params.args = [need_grad_tensors, grad_output, inner_args]
        self.data_params.kwargs = {}
        self.data_params.valid_input_index = 0
        self.data_params.origin_func = self.get_grad_input_from_vjp
        layer = LayerFactory.create(
            self.handler_params.api_name,
            self.handler_params.fuzz_device,
            self.handler_params.pert_mode,
        )
        layer.handle(self.data_params)
        self.perturbed_grad_input = tuple(
            [x.cpu() for x in self.data_params.perturbed_result]
        )
