# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import inspect
import os
from dataclasses import dataclass, is_dataclass
from typing import Tuple, Dict, Optional, Any
from functools import partial
import copy
from typing import Union

import numpy as np

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common.utils import convert_tuple, CompareException


@dataclass
class ModuleForwardInputsOutputs:
    args: Optional[Tuple]
    kwargs: Optional[Dict]
    output: Any

    @property
    def args_tuple(self):
        return convert_tuple(self.args)

    @property
    def output_tuple(self):
        return convert_tuple(self.output)

    def update_output_with_args_and_kwargs(self):
        self.output = self.args + tuple(self.kwargs.values())


@dataclass
class ModuleBackwardInputsOutputs:
    grad_output: Optional[Tuple]
    grad_input: Optional[Tuple]

    @property
    def grad_input_tuple(self):
        return convert_tuple(self.grad_input)

    @property
    def grad_output_tuple(self):
        return convert_tuple(self.grad_output)


@dataclass
class ModuleBackwardInputs:
    grad_input: Optional[Tuple]

    @property
    def grad_input_tuple(self):
        return convert_tuple(self.grad_input)


@dataclass
class ModuleBackwardOutputs:
    grad_output: Optional[Tuple]

    @property
    def grad_output_tuple(self):
        return convert_tuple(self.grad_output)


class TensorStatInfo:
    def __init__(self, max_val=None, min_val=None, mean_val=None, norm_val=None, stack_tensor_stat=None):
        self.max = max_val
        self.min = min_val
        self.mean = mean_val
        self.norm = norm_val
        self.stack_tensor_stat = stack_tensor_stat


class BaseDataProcessor:
    _recursive_key_stack = []
    special_type = (
        np.integer, np.floating, np.bool_, np.complexfloating, np.str_, np.byte, np.unicode_, np.ndarray,
        bool, int, float, str, slice,
        type(Ellipsis)
    )

    def __init__(self, config, data_writer):
        self.data_writer = data_writer
        self.config = config
        self.api_info_struct = {}
        self.stack_info_struct = {}
        self.current_api_or_module_name = None
        self.api_data_category = None
        self.current_iter = 0
        self._return_forward_new_output = False
        self._forward_new_output = None
        self.save_name = None
        if hasattr(config, "data_mode"):
            self.allowed_data_mode = self._get_allowed_data_mode(config.data_mode)

    @property
    def data_path(self):
        return self.data_writer.dump_tensor_data_dir

    @property
    def is_terminated(self):
        return False

    @staticmethod
    def analyze_api_call_stack(name):
        try:
            api_stack = inspect.stack()[5:]
        except Exception as e:
            logger.warning(f"The call stack of <{name}> failed to retrieve, {e}.")
            api_stack = None
        stack_str = []
        if api_stack:
            for (_, path, line, func, code, _) in api_stack:
                if not code:
                    continue
                stack_line = f"File {path}, line {str(line)}, in {func}, \n {code[0].strip()}"
                stack_str.append(stack_line)
        else:
            stack_str.append(Const.WITHOUT_CALL_STACK)
        stack_info_struct = {name: stack_str}
        return stack_info_struct

    @staticmethod
    def transfer_type(data):
        dtype = str(type(data))
        if 'int' in dtype:
            return int(data)
        elif 'float' in dtype:
            return float(data)
        else:
            return data

    @staticmethod
    def set_value_into_nested_structure(data_structure, indexes, value):
        '''
        Args:
            data_structure: nested data structure
            indexes: List
            value: value to be set
        '''
        if not indexes:
            raise ValueError("set_value_into_nested_structure failed: "
                             "indexes need to be non empty when set value to nested data structure")
        current_level = data_structure
        for i, index in enumerate(indexes):
            valid_for_list = isinstance(current_level, list) and isinstance(index, int) and len(current_level) > index
            valid_for_dict = isinstance(current_level, dict) and index in current_level
            is_last = i == len(indexes) - 1
            if valid_for_dict or valid_for_list:
                if is_last:
                    try:
                        current_level[index] = value
                    except Exception as e:
                        raise IndexError("set_value_into_nested_structure failed: passed indexes wrong") from e
                else:
                    try:
                        current_level = current_level[index]
                    except Exception as e:
                        raise IndexError("set_value_into_nested_structure failed: passed indexes wrong") from e
            else:
                raise ValueError("set_value_into_nested_structure failed: "
                                 "invalid data_structure type or invalid index")

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
    def _analyze_builtin(arg):
        single_arg = {}
        if isinstance(arg, slice):
            # The slice parameter may be of the tensor, numpy or other types.
            # It needs to be converted to the Python value type before JSON serialization
            single_arg.update({"type": "slice"})
            values = []
            for value in [arg.start, arg.stop, arg.step]:
                if value is not None:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(f"The data type {type(value)} cannot be converted to int type.")
                        value = None
                values.append(value)
            single_arg.update({"value": values})
        else:
            single_arg.update({"type": type(arg).__name__})
            # When arg is Ellipsis(...) type, it needs to be converted to str("...") type
            single_arg.update({"value": arg if arg is not Ellipsis else "..."})
        return single_arg

    @staticmethod
    def _analyze_numpy(ndarray, numpy_type):
        ndarray_json = {}
        ndarray_json.update({'type': 'numpy.ndarray'})
        ndarray_json.update({'dtype': str(ndarray.dtype)})
        ndarray_json.update({'shape': ndarray.shape})
        if ndarray.size > 0:
            ndarray_json.update({"Max": np.max(ndarray).item()})
            ndarray_json.update({"Min": np.min(ndarray).item()})
            ndarray_json.update({"Mean": np.mean(ndarray).item()})
            ndarray_json.update({"Norm": np.linalg.norm(ndarray).item()})
        else:
            ndarray_json.update({"Max": None})
            ndarray_json.update({"Min": None})
            ndarray_json.update({"Mean": None})
            ndarray_json.update({"Norm": None})
        return ndarray_json

    @staticmethod
    def _get_allowed_data_mode(data_mode):
        if Const.ALL in data_mode:
            allowed_data_mode = [Const.FORWARD, Const.BACKWARD, Const.INPUT, Const.OUTPUT]
        else:
            allowed_data_mode = list(set(data_mode))
            if Const.FORWARD not in allowed_data_mode and Const.BACKWARD not in allowed_data_mode:
                allowed_data_mode += [Const.FORWARD, Const.BACKWARD]
            if Const.INPUT not in allowed_data_mode and Const.OUTPUT not in allowed_data_mode:
                allowed_data_mode += [Const.INPUT, Const.OUTPUT]
        return allowed_data_mode

    @classmethod
    def get_special_types(cls):
        return cls.special_type

    @classmethod
    def recursive_apply_transform(cls, args, transform, depth=0) -> Union[dict, list, None]:
        if depth > Const.MAX_DEPTH:
            logger.error(f"The maximum depth of recursive transform, {Const.MAX_DEPTH} is reached.")
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        if isinstance(args, cls.get_special_types()):
            arg_transform = transform(args, cls._recursive_key_stack)
            return arg_transform
        elif isinstance(args, tuple) and hasattr(args, '_fields'):
            # namedtuple to dict
            args_dict = {field: getattr(args, field) for field in args._fields}
            return cls.apply_transform_dict(args_dict, transform, depth)
        elif is_dataclass(args):
            # dataclass to dict
            args_dict = {field: getattr(args, field) for field in args.__dataclass_fields__}
            return cls.apply_transform_dict(args_dict, transform, depth)
        elif isinstance(args, (list, tuple)):
            result_list = cls.apply_transform_list(args, transform, depth)
            return result_list
        elif isinstance(args, dict):
            return cls.apply_transform_dict(args, transform, depth)
        elif args is not None:
            logger.debug(f"Data type {type(args)} is not supported.")
            return None
        else:
            return None

    @classmethod
    def apply_transform_dict(cls, args, transform, depth):
        result_dict = {}
        for k, arg in args.items():
            cls._recursive_key_stack.append(k)
            result_dict[k] = cls.recursive_apply_transform(arg, transform, depth=depth + 1)
            cls._recursive_key_stack.pop()
        return result_dict

    @classmethod
    def apply_transform_list(cls, args, transform, depth):
        result_list = []
        for i, arg in enumerate(args):
            cls._recursive_key_stack.append(i)
            result_list.append(cls.recursive_apply_transform(arg, transform, depth=depth + 1))
            cls._recursive_key_stack.pop()
        return result_list

    @classmethod
    def register_hook_single_element(cls, element, suffix_stack, hook_fn):
        if cls.is_hookable_element(element):
            indexes = copy.deepcopy(suffix_stack)
            wrap_hook_fn = partial(hook_fn, indexes=indexes)

            def real_hook_fn(grad):
                return wrap_hook_fn(grad)
            element.register_hook(real_hook_fn)

    def if_return_forward_new_output(self):
        return self._return_forward_new_output

    def get_forward_new_output(self):
        self._return_forward_new_output = False
        return self._forward_new_output

    def update_iter(self, current_iter):
        self.current_iter = current_iter

    def update_api_or_module_name(self, api_or_module_name):
        if self.current_api_or_module_name != api_or_module_name:
            self.current_api_or_module_name = api_or_module_name

    def is_dump_for_data_mode(self, forward_backward, input_output):
        """
        Compare the parameters with data_mode to determine whether to dump.

        Args:
            forward_backward(str): The forward or backward mode to check.
            input_output(str): The input or output mode to check.

        Return:
            bool: True if the parameters are in data_mode or data_mode is all, False otherwise.
        """
        return forward_backward in self.allowed_data_mode and input_output in self.allowed_data_mode

    def analyze_element(self, element):
        return self.recursive_apply_transform(element, self.analyze_single_element)

    def analyze_forward_input(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        api_info_struct = {}
        # check whether data_mode contains forward or input
        if self.is_dump_for_data_mode(Const.FORWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT
            args_info_list = self.analyze_element(module_input_output.args_tuple)
            api_info_struct[name][Const.INPUT_ARGS] = args_info_list
            self.api_data_category = Const.KWARGS
            kwargs_info_list = self.analyze_element(module_input_output.kwargs)
            api_info_struct[name][Const.INPUT_KWARGS] = kwargs_info_list

        return api_info_struct

    def analyze_forward_output(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        api_info_struct = {}
        # check whether data_mode contains forward or input
        if self.is_dump_for_data_mode(Const.FORWARD, Const.OUTPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.OUTPUT
            output_info_list = self.analyze_element(module_input_output.output_tuple)
            api_info_struct[name][Const.OUTPUT] = output_info_list

        return api_info_struct

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        api_info_struct = {}
        # check whether data_mode contains forward or input
        if self.is_dump_for_data_mode(Const.FORWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT
            args_info_list = self.analyze_element(module_input_output.args_tuple)
            api_info_struct[name][Const.INPUT_ARGS] = args_info_list
            self.api_data_category = Const.KWARGS
            kwargs_info_list = self.analyze_element(module_input_output.kwargs)
            api_info_struct[name][Const.INPUT_KWARGS] = kwargs_info_list

        # check whether data_mode contains forward or output
        if self.is_dump_for_data_mode(Const.FORWARD, Const.OUTPUT):
            api_info_struct[name] = api_info_struct.get(name, {})
            self.api_data_category = Const.OUTPUT
            output_info_list = self.analyze_element(module_input_output.output_tuple)
            api_info_struct[name][Const.OUTPUT] = output_info_list

        if name in api_info_struct and hasattr(module_input_output, Const.PARAMS):
            self.api_data_category = Const.PARAMS
            api_info_struct[name][Const.PARAMS] = self.analyze_element(getattr(module_input_output, Const.PARAMS))

        return api_info_struct

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.BACKWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT
            input_info_list = self.analyze_element(module_input_output.grad_input_tuple)
            api_info_struct[name][Const.INPUT] = input_info_list

        if self.is_dump_for_data_mode(Const.BACKWARD, Const.OUTPUT):
            api_info_struct[name] = api_info_struct.get(name, {})
            self.api_data_category = Const.OUTPUT
            output_info_list = self.analyze_element(module_input_output.grad_output_tuple)
            api_info_struct[name][Const.OUTPUT] = output_info_list

        return api_info_struct

    def analyze_backward_input(self, name, module,
                               module_input_output: ModuleBackwardInputs):
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.BACKWARD, Const.INPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.INPUT

            input_info_list = self.analyze_element(module_input_output.grad_input_tuple)
            api_info_struct[name][Const.INPUT] = input_info_list
        return api_info_struct

    def analyze_backward_output(self, name, module,
                                module_input_output: ModuleBackwardOutputs):
        api_info_struct = {}
        if self.is_dump_for_data_mode(Const.BACKWARD, Const.OUTPUT):
            api_info_struct[name] = {}
            self.api_data_category = Const.OUTPUT

            output_info_list = self.analyze_element(module_input_output.grad_output_tuple)
            api_info_struct[name][Const.OUTPUT] = output_info_list
        return api_info_struct

    def analyze_params(self, name, param_name, grad):
        api_info_struct = {}
        self.save_name = name + Const.SEP + param_name
        data_info = self.analyze_element(grad)
        grad_info_dict = {param_name: [data_info]}
        api_info_struct[name] = grad_info_dict
        return api_info_struct

    def get_save_file_path(self, suffix):
        file_format = Const.PT_SUFFIX if self.config.framework == Const.PT_FRAMEWORK else Const.NUMPY_SUFFIX
        if self.save_name is not None:
            dump_data_name = (self.save_name + file_format)
            self.save_name = None
        else:
            dump_data_name = (self.current_api_or_module_name + Const.SEP + self.api_data_category + Const.SEP +
                              suffix + file_format)
        file_path = os.path.join(self.data_writer.dump_tensor_data_dir, dump_data_name)
        return dump_data_name, file_path

    def analyze_element_to_all_none(self, element):
        return self.recursive_apply_transform(element, lambda element, stack: None)

    def analyze_debug_forward(self, variable, name_with_count):
        self.current_api_or_module_name = name_with_count
        self.api_data_category = Const.TENSOR
        # these two attributes are used to construct tensor file name {name_with_count}.tensor.{indexes}.npy/pt
        data_info = self.analyze_element(variable)
        return data_info

    def analyze_debug_backward(self, variable, grad_name_with_count, nested_data_structure):
        def hook_fn(grad, indexes):
            suffix = Const.SEP.join([str(index) for index in indexes])
            self.save_name = grad_name_with_count + Const.SEP + Const.TENSOR + Const.SEP + suffix
            grad_data_info = self.analyze_element(grad)
            self.save_name = None
            full_index = [grad_name_with_count] + indexes
            try:
                self.set_value_into_nested_structure(nested_data_structure, full_index, grad_data_info)
            except (ValueError, IndexError) as e:
                logger.warning(f"error occured while recording statistics of {grad_name_with_count} variable, "
                               f"skip current recording, detailed infomation: {e}")
            return grad
        wrap_register_hook_single_element = partial(self.register_hook_single_element, hook_fn=hook_fn)
        self.recursive_apply_transform(variable, wrap_register_hook_single_element)