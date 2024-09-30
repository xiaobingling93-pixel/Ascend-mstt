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

import inspect
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Any

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

    def concat_args_and_kwargs(self):
        args = self.args + tuple(self.kwargs.values())
        return args


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
    def __init__(self, max_val=None, min_val=None, mean_val=None, norm_val=None):
        self.max = max_val
        self.min = min_val
        self.mean = mean_val
        self.norm = norm_val


class BaseDataProcessor:
    _recursive_key_stack = []
    special_type = (
        np.integer, np.floating, np.bool_, np.complexfloating, np.str_, np.byte, np.unicode_,
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
    def _analyze_numpy(value, numpy_type):
        return {"type": numpy_type, "value": value}

    @classmethod
    def get_special_types(cls):
        return cls.special_type

    @classmethod
    def recursive_apply_transform(cls, args, transform, depth=0):
        if depth > Const.MAX_DEPTH:
            logger.error(f"The maximum depth of recursive transform, {Const.MAX_DEPTH} is reached.")
            raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
        if isinstance(args, cls.get_special_types()):
            arg_transform = transform(args, cls._recursive_key_stack)
            return arg_transform
        elif isinstance(args, (list, tuple)):
            result_list = []
            for i, arg in enumerate(args):
                cls._recursive_key_stack.append(str(i))
                result_list.append(cls.recursive_apply_transform(arg, transform, depth=depth + 1))
                cls._recursive_key_stack.pop()
            return type(args)(result_list)
        elif isinstance(args, dict):
            result_dict = {}
            for k, arg in args.items():
                cls._recursive_key_stack.append(str(k))
                result_dict[k] = cls.recursive_apply_transform(arg, transform, depth=depth + 1)
                cls._recursive_key_stack.pop()
            return result_dict
        elif args is not None:
            logger.warning(f"Data type {type(args)} is not supported.")
            return None
        else:
            return None

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
        return (Const.ALL in self.config.data_mode or
                forward_backward in self.config.data_mode or
                input_output in self.config.data_mode)

    def analyze_pre_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        pass

    def analyze_element(self, element):
        return self.recursive_apply_transform(element, self.analyze_single_element)

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

    def get_save_file_path(self, suffix):
        file_format = Const.PT_SUFFIX if self.config.framework == Const.PT_FRAMEWORK else Const.NUMPY_SUFFIX
        dump_data_name = (self.current_api_or_module_name + Const.SEP + self.api_data_category + Const.SEP +
                          suffix + file_format)
        file_path = os.path.join(self.data_writer.dump_tensor_data_dir, dump_data_name)
        return dump_data_name, file_path
