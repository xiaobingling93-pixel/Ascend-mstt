import inspect
import os
import zlib
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import torch
try:
    import torch_npu
except ImportError:
    torch_npu = None


from ..common import recursive_apply_transform
from ..common.exceptions import MsaccException
from ..common.file_check import path_len_exceeds_limit, change_mode, FileCheckConst
from ..common.log import print_warn_log
from ..common.utils import Const
from ..free_benchmark import FreeBenchmarkCheck, UnequalRow

bits_for_overflow = 8


def build_data_processor(config, data_writer):
    if config.task == DataProcessor.full:
        return FullTensorDataProcessor(config, data_writer)
    elif config.task == DataProcessor.summary:
        return DataProcessor(config, data_writer)
    elif config.task == DataProcessor.overflow:
        return OverflowTensorDataProcessor(config, data_writer)
    elif config.task == DataProcessor.free_benchmark:
        return FreeBenchmarkDataProcessor(config, data_writer)
    else:
        raise MsaccException(MsaccException.INVALID_PARAM_ERROR,
                             "task should be in [{}, {}, {}, {}]".format(
                                 DataProcessor.full,
                                 DataProcessor.summary,
                                 DataProcessor.overflow,
                                 DataProcessor.free_benchmark
                             ))


@dataclass
class ModuleForwardInputsOutputs:
    args: Optional[Tuple]
    kwargs: Optional[Dict]
    output: Union[Tuple, torch.Tensor]

    @property
    def args_tuple(self):
        if not isinstance(self.args, tuple):
            return (self.args,)
        else:
            return self.args

    @property
    def output_tuple(self):
        if not isinstance(self.output, tuple):
            return (self.output,)
        else:
            return self.output

    def concat_args_and_kwargs(self):
        args = self.args + tuple(self.kwargs.values())
        return args


@dataclass
class ModuleBackwardInputsOutputs:
    grad_output: Optional[Tuple]
    grad_input: Optional[Tuple]

    @property
    def grad_input_tuple(self):
        if not isinstance(self.grad_input, tuple):
            return (self.grad_input,)
        else:
            return self.grad_input

    @property
    def grad_output_tuple(self):
        if not isinstance(self.grad_output, tuple):
            return (self.grad_output,)
        else:
            return self.grad_output


class TensorStatInfo:
    def __init__(self, max_val=None, min_val=None, mean_val=None, norm_val=None):
        self.max = max_val
        self.min = min_val
        self.mean = mean_val
        self.norm = norm_val


class DataProcessor:
    full = "tensor"
    summary = "statistics"
    overflow = "overflow_check"
    free_benchmark = "free_benchmark"

    def __init__(self, config, data_writer):
        self.data_writer = data_writer
        self.api_info_struct = {}
        self.stack_info_struct = {}
        self.torch_object_key = {
            "device": self.analyze_device_in_kwargs,
            "dtype": self.analyze_dtype_in_kwargs
        }
        self.current_api_or_module_name = None
        self.config = config
        self.api_data_category = None
        self.has_overflow = False
        self.current_iter = 0

        # 需要对forward的output进行更改
        self._return_forward_new_output = False
        self._forward_new_output = None

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

    def get_stat_info(self, data):
        tensor_stat = TensorStatInfo()
        if data.is_meta:
            return tensor_stat
        data_clone = data.detach()
        if data_clone.numel() == 0:
            return tensor_stat
        elif data_clone.dtype == torch.bool:
            tensor_stat.max = True in data_clone
            tensor_stat.min = False not in data_clone
            tensor_stat.mean = None
            tensor_stat.norm = None
        elif not data_clone.shape:
            tensor_stat.max = data_clone.item()
            tensor_stat.min = tensor_stat.max
            tensor_stat.mean = tensor_stat.max
            tensor_stat.norm = tensor_stat.max
        else:
            if not data_clone.is_floating_point():
                data_clone = data_clone.float()
            tensor_stat.max = torch._C._VariableFunctionsClass.max(data_clone).item()
            tensor_stat.min = torch._C._VariableFunctionsClass.min(data_clone).item()
            tensor_stat.mean = torch._C._VariableFunctionsClass.mean(data_clone).item()
            tensor_stat.norm = torch._C._VariableFunctionsClass.norm(data_clone).item()

        return tensor_stat

    def if_return_forward_new_output(self):
        return self._return_forward_new_output

    def get_forward_new_output(self):
        self._return_forward_new_output = False
        return self._forward_new_output

    def update_iter(self, current_iter):
        self.current_iter = current_iter

    def visit_and_clear_overflow_status(self, api_or_module_name):
        if self.current_api_or_module_name != api_or_module_name:
            self.current_api_or_module_name = api_or_module_name
            self.has_overflow = False

    def is_dump_for_data_mode(self, forward_backward, input_output):
        """
        Compare the parameters with data_mode to determine whether to dump.

        Args:
            forward_backward(str): The forward or backward mode to check.
            input_output(str): The input or output mode to check.

        Return:
            bool: True if the parameters are in data_mode or data_mode is all, False otherwise.
        """
        return (Const.ALL in self.config.data_mode or
                forward_backward in self.config.data_mode or
                input_output in self.config.data_mode)

    def analyze_single_element(self, element, suffix_stack):
        if suffix_stack and suffix_stack[-1] in self.torch_object_key:
            return self.torch_object_key[suffix_stack[-1]](element)

        if isinstance(element, torch.Size):
            return self._analyze_torch_size(element)

        converted_numpy, numpy_type = self._convert_numpy_to_builtin(element)
        if converted_numpy is not element:
            return self._analyze_numpy(converted_numpy, numpy_type)

        if isinstance(element, torch.Tensor):
            return self._analyze_tensor(element, Const.SEP.join(suffix_stack))

        if isinstance(element, (bool, int, float, str, slice)):
            return self._analyze_builtin(element)
        return {}

    def analyze_element(self, element):
        return recursive_apply_transform(element, self.analyze_single_element)

    def analyze_pre_forward(self, name, module,
                            module_input_output: ModuleForwardInputsOutputs):
        pass

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.FORWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT
            args_info_list = self.analyze_element(module_input_output.args_tuple)
            api_info_struct[name][Const.INPUT_ARGS] = args_info_list

            self.api_data_category = Const.KWARGS
            kwargs_info_list = self.analyze_element(module_input_output.kwargs)
            api_info_struct[name][Const.INPUT_KWARGS] = kwargs_info_list

        if self.is_dump_for_data_mode(Const.FORWARD, Const.OUTPUT):
            api_info_struct[name] = api_info_struct.get(name, {})
            self.api_data_category = Const.OUTPUT
            output_info_list = self.analyze_element(module_input_output.output_tuple)
            api_info_struct[name][Const.OUTPUT] = output_info_list

        return api_info_struct

    def analyze_pre_forward_inplace(self, name, module_input_output: ModuleForwardInputsOutputs):
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.FORWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT
            args_info_list = self.analyze_element(module_input_output.args_tuple)
            api_info_struct[name][Const.INPUT_ARGS] = args_info_list

            self.api_data_category = Const.KWARGS
            kwargs_info_list = self.analyze_element(module_input_output.kwargs)
            api_info_struct[name][Const.INPUT_KWARGS] = kwargs_info_list

        return api_info_struct

    def analyze_forward_inplace(self, name, module_input_output: ModuleForwardInputsOutputs):
        concat_args = module_input_output.concat_args_and_kwargs()
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.FORWARD, Const.OUTPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.OUTPUT
            output_info_list = self.analyze_element(concat_args)
            api_info_struct[name][Const.OUTPUT] = output_info_list

        return api_info_struct

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.BACKWARD, Const.OUTPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.OUTPUT
            input_info_list = self.analyze_element(module_input_output.grad_input_tuple)
            api_info_struct[name][Const.GRAD_INPUT] = input_info_list

        if self.is_dump_for_data_mode(Const.BACKWARD, Const.INPUT):
            api_info_struct[name] = api_info_struct.get(name, {})
            self.api_data_category = Const.INPUT
            output_info_list = self.analyze_element(module_input_output.grad_output_tuple)
            api_info_struct[name][Const.GRAD_OUTPUT] = output_info_list

        return api_info_struct

    def _analyze_numpy(self, value, numpy_type):
        single_arg = {}
        single_arg.update({"type": numpy_type})
        single_arg.update({"value": value})
        return single_arg

    def _analyze_builtin(self, arg):
        single_arg = {}
        if isinstance(arg, slice):
            single_arg.update({"type": "slice"})
            # slice参数中可能存在tensor类型，json序列化，需要转换为python数值类型
            values = [
                value if not isinstance(value, torch.Tensor) else value.item()
                for value in [arg.start, arg.stop, arg.step]
            ]
            single_arg.update({"value": values})
        else:
            single_arg.update({"type": type(arg).__name__})
            single_arg.update({"value": arg})
        return single_arg

    def _analyze_torch_size(self, arg):
        single_arg = {}
        single_arg.update({"type": "torch.Size"})
        single_arg.update({"value": list(arg)})
        return single_arg

    def _analyze_maybe_overflow_tensor(self, tensor_json, tensor):
        data_clone = tensor.detach()
        if hasattr(torch_npu._C, '_npu_is_support_inf_nan') and torch_npu._C._npu_is_support_inf_nan():
            if tensor_json[Const.MAX] is None:
                return
            if np.isinf(tensor_json[Const.MAX]) or np.isnan(tensor_json[Const.MAX]):
                tensor_json['Max_except_inf_nan'] = self.handle_tensor_extremum_nan_inf(data_clone, "max")
                self.has_overflow = True
            if np.isinf(tensor_json[Const.MIN]) or np.isnan(tensor_json[Const.MIN]):
                tensor_json['Min_except_inf_nan'] = self.handle_tensor_extremum_nan_inf(data_clone, "min")
                self.has_overflow = True
        else:
            self.has_overflow = check_overflow_npu()
            if self.has_overflow:
                clear_overflow_npu()

    def _analyze_tensor(self, tensor, suffix):
        tensor_stat = self.get_stat_info(tensor)

        tensor_json = {}
        tensor_json.update({'type': 'torch.Tensor'})
        tensor_json.update({'dtype': str(tensor.dtype)})
        tensor_json.update({"shape": tensor.shape})
        tensor_json.update({"Max": tensor_stat.max})
        tensor_json.update({"Min": tensor_stat.min})
        self._analyze_maybe_overflow_tensor(tensor_json, tensor)
        tensor_json.update({"Mean": tensor_stat.mean})
        tensor_json.update({"Norm": tensor_stat.norm})
        tensor_json.update({"requires_grad": tensor.requires_grad})
        if self.config.summary_mode == "md5":
            tensor_md5 = self.get_md5_for_tensor(tensor)
            tensor_json.update({"md5": tensor_md5})

        return tensor_json


class FullTensorDataProcessor(DataProcessor):

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.data_path = self.data_writer.dump_tensor_data_dir

    def _analyze_tensor(self, tensor, suffix):
        dump_data_name = (self.current_api_or_module_name + Const.SEP + self.api_data_category + Const.SEP +
                          suffix + ".pt")
        file_path = os.path.join(self.data_writer.dump_tensor_data_dir, dump_data_name)
        if not path_len_exceeds_limit(file_path):
            torch.save(tensor, file_path)
            change_mode(file_path, FileCheckConst.DATA_FILE_AUTHORITY)
        else:
            print_warn_log(f'The file path {file_path} length exceeds limit.')
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        return single_arg


class OverflowTensorDataProcessor(DataProcessor):
    __slots__ = ["cached_tensors_and_file_paths"]

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.cached_tensors_and_file_paths = {}
        self.real_overflow_dump_times = 0
        self.overflow_nums = config.overflow_num

    def _analyze_tensor(self, tensor, suffix):
        dump_data_name = (self.current_api_or_module_name + Const.SEP + self.api_data_category + Const.SEP +
                          suffix + ".pt")
        file_path = os.path.join(self.data_writer.dump_tensor_data_dir, dump_data_name)
        if not path_len_exceeds_limit(file_path):
            self.cached_tensors_and_file_paths.update({file_path: tensor})
        else:
            print_warn_log(f'The file path {file_path} length exceeds limit.')
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        return single_arg

    def analyze_forward(self, name, module,
                        module_input_output: ModuleForwardInputsOutputs):
        self.has_overflow = False
        api_info_struct = super().analyze_forward(name, module, module_input_output)
        self.maybe_save_overflow_data_and_check_overflow_times()
        return api_info_struct if self.has_overflow else None

    def analyze_backward(self, name, module,
                         module_input_output: ModuleBackwardInputsOutputs):
        self.has_overflow = False
        api_info_struct = super().analyze_backward(name, module, module_input_output)
        self.maybe_save_overflow_data_and_check_overflow_times()
        return api_info_struct if self.has_overflow else None

    def maybe_save_overflow_data_and_check_overflow_times(self):
        if self.has_overflow:
            for file_path, tensor in self.cached_tensors_and_file_paths.items():
                torch.save(tensor, file_path)
                change_mode(file_path, FileCheckConst.DATA_FILE_AUTHORITY)
            self.inc_and_check_overflow_times()
        self.cached_tensors_and_file_paths = {}

    def inc_and_check_overflow_times(self):
        self.real_overflow_dump_times += 1
        if self.overflow_nums == -1:
            return
        if self.real_overflow_dump_times >= self.overflow_nums:
            raise MsaccException(MsaccException.OVERFLOW_NUMS_ERROR,
                                 str(self.real_overflow_dump_times))


class FreeBenchmarkDataProcessor(DataProcessor):

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.checker = FreeBenchmarkCheck(config=config)

    def update_iter(self, current_iter):
        self.current_iter = current_iter
        self.checker.update_iter(current_iter)

    def update_unequal_rows(self, unequal_rows: List[UnequalRow]):
        if len(unequal_rows) == 0:
            return
        for row in unequal_rows:
            data_dict = asdict(row)
            self.data_writer.write_data_to_csv(
                data_dict.values(),
                data_dict.keys(),
                self.data_writer.free_benchmark_file_path
            )
        return

    def analyze_pre_forward(self, name, module,
                            module_input_output: ModuleForwardInputsOutputs):
        args = module_input_output.args
        kwargs = module_input_output.kwargs
        self.checker.pre_forward(name, module, self, args, kwargs)

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        new_output, unequal_rows = self.checker.forward(
            name,
            module,
            module_input_output.args,
            module_input_output.kwargs,
            module_input_output.output,
        )
        self.update_unequal_rows(unequal_rows)
        if self.checker.if_fix():
            self._return_forward_new_output = True
            self._forward_new_output = new_output

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        self.checker.backward(name, module, module_input_output.grad_output)


def overflow_debug_mode_enable():
    overflow_mode = os.getenv(OverflowConst.OVERFLOW_DEBUG_MODE_ENABLE, Const.ENV_DISABLE)
    return overflow_mode == Const.ENV_ENABLE


def check_overflow_npu():
    if overflow_debug_mode_enable():
        float_status = torch.zeros(bits_for_overflow).npu()
        result = torch_npu.npu_get_float_status(float_status, OverflowConst.OVERFLOW_DEBUG_MODE)
        if (result.cpu()[0] != 0):
            return True
        else:
            return False
    else:
        return torch_npu._C._check_overflow_npu()


def clear_overflow_npu():
    if overflow_debug_mode_enable():
        float_status = torch.zeros(bits_for_overflow).npu()
        torch_npu.npu_clear_float_status(float_status, OverflowConst.OVERFLOW_DEBUG_MODE)
    else:
        torch_npu._C._clear_overflow_npu()


class OverflowConst:
    """
    Class for Overflow
    """
    OVERFLOW_DEBUG_MODE_ENABLE = "OVERFLOW_DEBUG_MODE_ENABLE"
    OVERFLOW_ORIGINAL_MODE = 0
    OVERFLOW_DEBUG_MODE = 1
