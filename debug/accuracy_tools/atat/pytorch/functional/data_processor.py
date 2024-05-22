import torch
import zlib
import numpy as np
import os
import inspect
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Union
from ..common.exceptions import MsaccException
from ..common.utils import Const
from ..common import recursive_apply_transform


def build_data_processor(task, task_config, data_writer):
    if task == DataProcessor.full:
        return FullTensorDataProcessor(task_config, data_writer)
    elif task == DataProcessor.summary:
        return DataProcessor(task_config, data_writer)
    elif task == DataProcessor.overflow:
        return OverflowTensorDataProcessor(task_config, data_writer)
    else:
        raise MsaccException(MsaccException.INVALID_PARAM_ERROR,
                                  "task should be in [{}, {}, {}]".format(
                                      DataProcessor.full,
                                      DataProcessor.summary,
                                      DataProcessor.overflow
                                  ))


@dataclass
class ModuleForwardInputsOutputs:
    args: Optional[Tuple]
    kwargs: Optional[Dict]
    output: Union[Tuple, torch.Tensor]

    def __init__(self, args, kwargs, output):
        if not isinstance(args, tuple):
            args = (args, )
        if not isinstance(output, tuple):
            output = (output, )
        self.args = args
        self.kwargs = kwargs
        self.output = output


@dataclass
class ModuleBackwardInputsOutputs:
    grad_output: Optional[Tuple]
    grad_input: Optional[Tuple]

    def __init__(self, grad_input, grad_output):
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input, )
        if not isinstance(grad_output, tuple):
            grad_output = (grad_output,)
        self.grad_input = grad_input
        self.grad_output = grad_output


class DataProcessor:
    full = "tensor"
    summary = "summary"
    overflow = "overflow"

    def __init__(self, task_config, data_writer):
        self.data_writer = data_writer
        self.api_info_struct = {}
        self.stack_info_struct = {}
        self.torch_object_key = {
            "device": self.analyze_device_in_kwargs,
            "dtype": self.analyze_dtype_in_kwargs
        }
        self.api_name = None
        self.task_config = task_config
        self.api_data_category = None
        self.has_overflow = False

    @staticmethod
    def get_md5_for_tensor(x):
        if x.dtype == torch.bfloat16:
            x = x.float()
        tensor_bytes = x.cpu().detach().numpy().tobytes()
        crc32_hash = zlib.crc32(tensor_bytes)
        return f"{crc32_hash:08x}"

    @staticmethod
    def analyze_device_in_kwargs(element):
        single_arg = {}
        single_arg.update({'type': "torch.device"})
        if not isinstance(element, str):
            if hasattr(element, "index"):
                device_value = element.type + ":" + str(element.index)
            else:
                device_value = element.type
            single_arg.update({"value": device_value})
        else:
            single_arg.update({"value": element})
        return single_arg

    @staticmethod
    def analyze_dtype_in_kwargs(element):
        single_arg = {}
        single_arg.update({"type": "torch.dtype"})
        single_arg.update({"value": str(element)})
        return single_arg

    @staticmethod
    def _convert_numpy_to_builtin(arg):
        type_mapping = {
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
            np.complexfloating: complex,
            np.str_: str,
            np.byte: bytes,
            np.unicode_: str
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(arg, numpy_type):
                return builtin_type(arg), type(arg).__name__
        return arg, ''

    def _analyze_numpy(self, value, numpy_type):
        single_arg = {}
        single_arg.update({"type": numpy_type})
        single_arg.update({"value": value})
        return single_arg

    def get_stat_info(self, data):
        if data.is_meta:
            return
        data_clone = data.detach()
        if data_clone.numel() == 0:
            tensor_max = None
            tensor_min = None
            tensor_mean = None
            tensor_norm = None
        elif data_clone.dtype == torch.bool:
            tensor_max = True in data_clone
            tensor_min = False not in data_clone
            tensor_mean = None
            tensor_norm = None
        elif not len(data_clone.shape):
            tensor_max = data_clone.item()
            tensor_min = tensor_max
            tensor_mean = tensor_max
            tensor_norm = tensor_max
        else:
            if not data_clone.is_floating_point():
                data_clone = data_clone.float()
            tensor_max = torch._C._VariableFunctionsClass.max(data_clone).item()
            tensor_min = torch._C._VariableFunctionsClass.min(data_clone).item()
            tensor_mean = torch._C._VariableFunctionsClass.mean(data_clone).item()
            tensor_norm = torch._C._VariableFunctionsClass.norm(data_clone).item()

        return tensor_max, tensor_min, tensor_mean, tensor_norm

    def _analyze_builtin(self, arg):
        single_arg = {}
        if isinstance(arg, slice):
            single_arg.update({"type": "slice"})
            single_arg.update({"value": [arg.start, arg.stop, arg.step]})
        else:
            single_arg.update({"type": type(arg).__name__})
            single_arg.update({"value": arg})
        return single_arg

    @staticmethod
    def handle_tensor_extremum_nan_inf(data_clone, operator):
        data_nan = torch._C._VariableFunctionsClass.isnan(data_clone)
        if int(torch._C._VariableFunctionsClass.sum(data_nan)) == data_clone.numel():
            return float('nan')
        finite_mask = torch._C._VariableFunctionsClass.isfinite(data_clone)
        if int(torch._C._VariableFunctionsClass.sum(finite_mask)) > 0:
            finite_values = data_clone[finite_mask]
            return torch._C._VariableFunctionsClass.max(finite_values).item() if operator == 'max' else \
                torch._C._VariableFunctionsClass.min(finite_values).item()
        else:
            data_no_nan = data_clone[~data_nan]
            return torch._C._VariableFunctionsClass.max(data_no_nan).item() if operator == 'max' else \
                torch._C._VariableFunctionsClass.min(data_no_nan).item()

    def _analyze_maybe_overflow_tensor(self, tensor_json, tensor):
        if np.isinf(tensor_json['Max']) or np.isnan(tensor_json['Max']):
            tensor_json['Max_except_inf_nan'] = self.handle_tensor_extremum_nan_inf(tensor, "max")
            self.has_overflow = True
        if np.isinf(tensor_json['Min']) or np.isnan(tensor_json['Min']):
            tensor_json['Min_except_inf_nan'] = self.handle_tensor_extremum_nan_inf(tensor, "min")
            self.has_overflow = True

    def _analyze_tensor(self, tensor, suffix):
        tensor_max, tensor_min, tensor_mean, tensor_norm = self.get_stat_info(tensor)

        tensor_json = {}
        tensor_json.update({'type': 'torch.Tensor'})
        tensor_json.update({'dtype': str(tensor.dtype)})
        tensor_json.update({"shape": tensor.shape})
        tensor_json.update({"Max": tensor_max})
        tensor_json.update({"Min": tensor_min})
        self._analyze_maybe_overflow_tensor(tensor_json, tensor)
        tensor_json.update({"Mean": tensor_mean})
        tensor_json.update({"Norm": tensor_norm})
        tensor_json.update({"requires_grad": tensor.requires_grad})
        if self.task_config.md5:
            tensor_md5 = self.get_md5_for_tensor(tensor)
            tensor_json.update({"md5": tensor_md5})

        return tensor_json

    def analyze_single_element(self, element, suffix_stack):
        if suffix_stack and suffix_stack[-1] in self.torch_object_key:
            return self.torch_object_key[suffix_stack[-1]](element)

        converted_numpy, numpy_type = self._convert_numpy_to_builtin(element)
        if converted_numpy is not element:
            return self._analyze_numpy(converted_numpy, numpy_type)

        if isinstance(element, torch.Tensor):
            return self._analyze_tensor(element, Const.SEP.join(suffix_stack))

        if isinstance(element, (bool, int, float, str, slice)):
            return self._analyze_builtin(element)

    def analyze_element(self, element):
        return recursive_apply_transform(element, self.analyze_single_element)

    @staticmethod
    def analyze_api_call_stack(name):
        stack_str = []
        for (_, path, line, func, code, _) in inspect.stack()[5:]:
            if not code:
                continue
            stack_line = " ".join([
                "File", ", ".join([
                    path,
                    " ".join(["line", str(line)]),
                    " ".join(["in", func]),
                    " ".join(["\n", code[0].strip()])
                ])
            ])
            stack_str.append(stack_line)
        stack_info_struct = {name: stack_str}
        return stack_info_struct

    def analyze_forward(self, name,
                        module_input_output: ModuleForwardInputsOutputs):
        self.api_name = name
        self.api_data_category = "input"
        args_info_list = self.analyze_element(module_input_output.args)
        self.api_data_category = "kwargs"
        kwargs_info_list = self.analyze_element(module_input_output.kwargs)
        self.api_data_category = "output"
        output_info_list = self.analyze_element(module_input_output.output)
        api_info_struct = {name: {"input_args": args_info_list,
                                  "input_kwargs": kwargs_info_list,
                                  "output": output_info_list}}
        return api_info_struct

    def analyze_backward(self, name,
                         module_input_output: ModuleBackwardInputsOutputs):
        self.api_name = name
        self.api_data_category = "output"
        input_info_list = self.analyze_element(module_input_output.grad_input)
        self.api_data_category = "input"
        output_info_list = self.analyze_element(module_input_output.grad_output)
        api_info_struct = {name: {"grad_input": input_info_list, "grad_output": output_info_list}}  # TODO: magic str
        return api_info_struct


class FullTensorDataProcessor(DataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        self.data_path = self.data_writer.dump_tensor_data_dir
        dump_data_name = (self.api_name + Const.SEP + self.api_data_category + Const.SEP +
                          suffix + ".pt")
        file_path = os.path.join(self.data_writer.dump_tensor_data_dir, dump_data_name)
        torch.save(tensor, file_path)
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        return single_arg


class OverflowTensorDataProcessor(FullTensorDataProcessor):
    __slots__ = ["cached_tensors_and_file_paths"]

    def __init__(self, task_config, data_writer):
        super().__init__(task_config, data_writer)
        self.cached_tensors_and_file_paths = {}

    def _analyze_tensor(self, tensor, suffix):
        self.data_path = self.data_writer.dump_tensor_data_dir
        dump_data_name = (self.api_name + Const.SEP + self.api_data_category + Const.SEP +
                          suffix + ".pt")
        file_path = os.path.join(self.data_writer.dump_tensor_data_dir, dump_data_name)
        self.cached_tensors_and_file_paths.update({file_path: tensor})
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})

    def analyze_forward(self, name,
                        module_input_output: ModuleForwardInputsOutputs):
        self.has_overflow = False
        api_info_struct = super().analyze_forward(name, module_input_output)
        if self.has_overflow:
            self.save_overflow_data()
            return api_info_struct
        return None

    def analyze_backward(self, name,
                        module_input_output: ModuleBackwardInputsOutputs):
        self.has_overflow = False
        api_info_struct = super().analyze_backward(name, module_input_output)
        if self.has_overflow:
            self.save_overflow_data()
            return api_info_struct
        return None

    def save_overflow_data(self):
        for file_path, tensor in self.cached_tensors_and_file_paths.items():
            torch.save(tensor, file_path)
        self.cached_tensors_and_file_paths = {}
