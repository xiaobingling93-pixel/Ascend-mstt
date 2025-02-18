#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import argparse
import json
import os
import re

import math
import numpy as np
import torch

from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import binary_standard_api, absolute_standard_api, \
ulp_standard_api, thousandth_standard_api
from msprobe.core.common.file_utils import FileOpen, load_json, save_json
from msprobe.core.common.utils import check_file_or_directory_path, check_op_str_pattern_valid, is_int
from msprobe.core.common.const import Const, MonitorConst, MsgConst
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import make_dir
from msprobe.core.common.utils import recursion_depth_decorator

TENSOR_DATA_LIST = ["torch.Tensor", "torch.nn.parameter.Parameter"]
TORCH_BOOL_TYPE = ["torch.bool"]
TORCH_INT_TYPE = ["torch.uint8", "torch.int8", "torch.int16", "torch.short", "torch.int32", "torch.int",
                  "torch.int64", "torch.long"]
TORCH_FLOAT_TYPE = ["torch.float16", "torch.half", "torch.bfloat16", "torch.float32", "torch.float",
                    "torch.float64", "torch.double"]
TORCH_COMPLEX_TYPE = ["torch.complex32", "torch.chalf", "torch.complex64", "torch.cfloat", "torch.complex128",
                      "torch.cdouble"]
OPERATOR_TYPE = ("Functional", "Tensor", "Torch")

API_INFO = 2
FOUR_SEGMENT = 4
FIVE_SEGMENT = 5
DATA_NAME = "data_name"
API_MAX_LENGTH = 30
PROPAGATION_LIST = [Const.FORWARD, Const.BACKWARD]
DATAMODE_LIST = ["random_data", "real_data"]


class APIInfo:
    def __init__(self, api_full_name, api_info_dict, backward_info=None):
        self.api_full_name = api_full_name
        self.api_info_dict = api_info_dict
        self.backward_info = backward_info

    @property
    def api_type(self):
        return self.api_full_name.split(Const.SEP, -1)[0]

    @classmethod
    def from_json(cls, json_content, propagation):
        forward_name, forward_dict = list(json_content.items())[0]
        forward_info = cls(api_full_name=forward_name, api_info_dict=forward_dict)

        if propagation == Const.BACKWARD:
            backward_name, backward_dict = list(json_content.items())[1]
            backward_info = cls(api_full_name=backward_name, api_info_dict=backward_dict)
            forward_info.backward_info = backward_info

        if not forward_info.is_supported_type():
            raise ValueError(f"type {forward_info.api_type} of API is not supported!")

        return forward_info

    def is_supported_type(self):
        return self.api_type in OPERATOR_TYPE


class CommonConfig:
    def __init__(self, json_config):
        self.dump_json_path = json_config.get('dump_json_path')
        self.api_name = json_config.get('api_name')
        self.extract_api_path = json_config.get('extract_api_path')
        self.propagation = json_config.get('propagation')
        self.data_mode = json_config.get('data_mode')
        self.random_seed = json_config.get('random_seed')
        self.iter_times = json_config.get('iter_times')
        self._check_config()


    def check_user_settings(self):
        iter_t = self.iter_times
        if iter_t <= 0:
            raise ValueError("iter_times should be an integer bigger than zero!")

        json_file = self.extract_api_path
        propagation = self.propagation

        json_content = load_json(json_file)

        # ensure the dict is not empty
        if not json_content:
            raise ValueError(f'json file is empty!')

        # ensure json_content is of type dict
        if not isinstance(json_content, dict):
            raise ValueError(f'content of json file is not a dict!')

        # ensure the length of json_content is within allowed limits
        if len(json_content) > API_INFO:
            raise ValueError(f'json file has more than one API, the API only contains forward and backward info')

        # Retrieve the first API name and dictionary
        forward_item = next(iter(json_content.items()), None)
        if not forward_item or not isinstance(forward_item[1], dict):
            raise ValueError(f'Invalid forward API data in json_content!')

        # if propagation is backward, ensure json file contains forward and backward info
        if propagation == Const.BACKWARD and len(json_content) < API_INFO:
            raise ValueError(f'Backward propagation requires contains forward and backward info!')

        # if propagation is backward, ensure it has valid data
        if propagation == Const.BACKWARD:
            backward_item = list(json_content.items())[1]
            if not isinstance(backward_item[1], dict):
                raise ValueError(f'Invalid backward API data in json_content!')

        return json_content


    def _check_config(self):
        if self.dump_json_path:
            check_file_or_directory_path(self.dump_json_path)
        if self.api_name:
            check_op_str_pattern_valid(self.api_name)
            if len(self.api_name) > API_MAX_LENGTH:
                raise ValueError(f'API name {self.api_name} is too long!')
        make_dir(os.path.dirname(self.extract_api_path))
        if self.propagation and self.propagation not in PROPAGATION_LIST:
            raise ValueError(f'propagation is invalid, it should be one of {PROPAGATION_LIST}')
        if self.data_mode and self.data_mode not in DATAMODE_LIST:
            raise ValueError(f'data_mode is invalid, it should be one of {DATAMODE_LIST}')
        if not is_int(self.random_seed):
            raise ValueError(f'random_seed is invalid, it should be an int')
        if not is_int(self.iter_times):
            raise ValueError(f'iter_times is invalid, it should be an int')


class APIExtractor:
    def __init__(self, api_name, dump_json_path, output_file):
        self.api_name = api_name
        self.dump_json_path = dump_json_path
        self.output_file = output_file
        self.data = None

    def extract_op(self):
        self.data = load_json(self.dump_json_path)
        new_data = {}
        extract_key_pattern = re.compile(f"^{re.escape(self.api_name)}\..+")
        real_data_path = self.data.get('dump_data_dir', '')
        for key, value in self.data.get('data', {}).items():
            if extract_key_pattern.match(key):
                if real_data_path:
                    value = self.load_real_data_path(value, real_data_path)
                new_data[key] = value
        if not new_data:
            logger.error(f"Error: The api '{self.api_name}' does not exist in the file.")
        else:
            save_json(self.output_file, new_data, indent=4)
            logger.info(
                f"The api '{self.api_name}' has been successfully extracted and saved in: {self.output_file}")

    def load_real_data_path(self, value, dump_data_dir):
        parameters = [Const.INPUT_ARGS, Const.GRAD_INPUT, Const.INPUT, Const.OUTPUT, Const.GRAD_OUTPUT]
        for parameter in parameters:
            for v in value.get(parameter, []):
                if v is not None:
                    self.update_data_name(v, dump_data_dir)
        return value

    def update_data_name(self, data, dump_data_dir):
        if isinstance(data, list):
            for item in data:
                self.update_data_name(item, dump_data_dir)
        elif DATA_NAME in data:
            data[DATA_NAME] = os.path.join(dump_data_dir, data[DATA_NAME])


class OperatorScriptGenerator:
    def __init__(self, common_config, args_info_forward, kwargs_info_forward, args_info_backward):
        self.common_config = common_config
        self.args_info_forward = args_info_forward
        self.kwargs_info_forward = kwargs_info_forward
        self.args_info_backward = args_info_backward

    @staticmethod
    def get_compare_standard(api_name):
        api_standard_map = {
            "binary_standard_api": "CompareStandard.BINARY_EQUALITY_STANDARD",
            "absolute_standard_api": "CompareStandard.ABSOLUTE_THRESHOLD_STANDARD",
            "ulp_standard_api": "CompareStandard.ULP_ERROR_STANDARD",
            "thousandth_standard_api": "CompareStandard.THOUSANDTH_STANDARD"
        }
        for standard_api, standard_value in api_standard_map.items():
            if api_name in globals()[standard_api]:
                return standard_value
        return "CompareStandard.BENCHMARK_STANDARD"

    @staticmethod
    def extract_detailed_api_segments(full_api_name):
        """
        Function Description:
            Extract the name of the API.
        Parameter:
            full_api_name_with_direction_status: Full name of the API. Example: torch.matmul.0.forward.output.0
        Return:
            api_name: Name of api. Example: matmul, mul, etc.
            full_api_name: Full name of api. Example: torch.matmul.0
            direction_status: Direction status of api. Example: forward, backward, etc.
        """
        api_parts = full_api_name.split(Const.SEP)
        api_parts_length = len(api_parts)
        api_type, api_name, api_order = None, None, None
        if api_parts_length == FOUR_SEGMENT:
            api_type, api_name, api_order, _ = api_parts
        elif api_parts_length == FIVE_SEGMENT:
            api_type, prefix, api_name, api_order, _ = api_parts
            api_name = Const.SEP.join([prefix, api_name])
        return api_type, api_name, api_order

    def get_settings(self, api_full_name):
        '''
        internal_settings contain all information needed for the operator program.
        keys:
            api_full_name: api_type.api_name.ordinal_number
            api_type: type of API, one of torch.nn.functional, torch.Tensor or Torch
            api_name: name of API
            ordinal_number: how many times the same api has been called
            direction_status: forward
            random_seed: if mode is random_data, random seed is random_seed
            iter_times: if mode is random_data, generate iter_times group of data; if mode is real_data,
            iter_times does not matter
            args_element_assignment: code for args assignment
            args_list_generator_device: code for generate args list on device
            args_list_generator_bench: code for generate args list on bench
            kwargs_value_assignment: code for kwargs assignment
            kwargs_dict_generator_device: code for generate kwargs dict on device
            kwargs_dict_generator_bench: code for generate kwargs dict on bench
        '''
        # Generate an internal setting dictionary based on user settings
        # including API name, type, comparison standard, random seed, number of iterations and other information
        internal_settings = {}
        internal_settings["propagation"] = self.common_config.propagation
        internal_settings["api_full_name"] = api_full_name
        api_type, api_name, ordinal_number = self.extract_detailed_api_segments(api_full_name)
        if api_type == "Functional":
            internal_settings["api_type"] = "torch.nn.functional"
        elif api_type == "Tensor":
            internal_settings["api_type"] = "torch.Tensor"
        else:
            internal_settings["api_type"] = "torch"
        internal_settings["api_name"] = api_name
        internal_settings["compare_standard"] = self.get_compare_standard(api_name)
        internal_settings["ordinal_number"] = ordinal_number
        internal_settings["direction_status"] = self.common_config.propagation
        internal_settings["random_seed"] = self.common_config.random_seed
        if self.common_config.data_mode == "real_data":
            internal_settings["iter_times"] = 1
        else:
            internal_settings["iter_times"] = self.common_config.iter_times
        internal_settings["args_element_assignment"] = \
                            self.generate_args_element_assignment_code(self.args_info_forward)
        internal_settings["args_list_generator_device"] = \
                            self.generate_args_list(self.args_info_forward, flag_device=True)
        internal_settings["args_list_generator_bench"] = \
                            self.generate_args_list(self.args_info_forward, flag_device=False)
        internal_settings["kwargs_value_assignment"] = \
                            self.generate_kwargs_value_assignment_code(self.kwargs_info_forward)
        internal_settings["kwargs_dict_generator_device"] = \
                            self.generate_kwargs_dict(self.kwargs_info_forward, flag_device=True)
        internal_settings["kwargs_dict_generator_bench"] = \
                            self.generate_kwargs_dict(self.kwargs_info_forward, flag_device=False)
        if self.common_config.propagation == Const.BACKWARD:
            internal_settings["args_element_assignment_backward"] = self.generate_args_element_assignment_code(
                self.args_info_backward)
            internal_settings["args_list_generator_device_backward"] = \
                            self.generate_args_list(self.args_info_backward, flag_device=True)
            internal_settings["args_list_generator_bench_backward"] = \
                            self.generate_args_list(self.args_info_backward, flag_device=False)
        else:
            internal_settings["args_element_assignment_backward"] = ''
            internal_settings["args_list_generator_device_backward"] = ''
            internal_settings["args_list_generator_bench_backward"] = ''

        return internal_settings

    @recursion_depth_decorator("OpGenerator: OperatorScriptGenerator.recursive_args_element_assignment")
    def recursive_args_element_assignment(self, args_info, name_number):
        args_element_assignment = ""
        for index, arg in enumerate(args_info):
            if isinstance(arg, (list, tuple)):
                new_args_element_assignment = \
                    self.recursive_args_element_assignment(arg, name_number + "_" + str(index))
                args_element_assignment += new_args_element_assignment
            else:
                arg["parameter_name"] = "arg" + name_number + "_" + str(index)
                args_element_assignment += "    " + "arg_info" + name_number + "_" + str(index) + " = " + \
                    "{}".format(str(arg)) + MsgConst.SPECIAL_CHAR[0]
                args_element_assignment += "    " + "arg" + name_number + "_" + str(index) + " = " + \
                    "generate_data(arg_info" + name_number + "_" + str(index) + ")" + MsgConst.SPECIAL_CHAR[0]
        return args_element_assignment


    def generate_args_element_assignment_code(self, args_info):
        args_element_assignment = self.recursive_args_element_assignment(args_info, "")
        return args_element_assignment

    @recursion_depth_decorator("OpGenerator: OperatorScriptGenerator.recursive_args_list")
    def recursive_args_list(self, args_info, flag_device=False, flag_bench=False):
        args_list_generator = ""
        for _, arg in enumerate(args_info):
            if isinstance(arg, (list, tuple)):
                (left_bracket, right_bracket) = ("[", "]") if isinstance(arg, list) else ("(", ")")
                args_list_generator += left_bracket
                new_args_list_generator = self.recursive_args_list(arg, flag_device=flag_device, flag_bench=flag_bench)
                args_list_generator += new_args_list_generator
                args_list_generator += right_bracket
            else:
                args_list_generator += arg.get("parameter_name")
                if arg.get("type") in TENSOR_DATA_LIST:
                    if flag_device:
                        args_list_generator += ".to(device)"
                    if flag_bench:
                        args_list_generator += '.to(torch.device("cpu"))'
                        args_list_generator += ".to(RAISE_PRECISION.get(str(" + arg.get("parameter_name") + \
                            ".dtype), " + arg.get("parameter_name") + ".dtype))"
            args_list_generator += Const.COMMA
        return args_list_generator

    def generate_args_list(self, args_info, flag_device):
        if flag_device:
            args_list_generator = self.recursive_args_list(args_info, flag_device=True)
        else:
            args_list_generator = self.recursive_args_list(args_info, flag_bench=True)
        return args_list_generator

    @recursion_depth_decorator("OpGenerator: OperatorScriptGenerator.recursive_kwargs_value_assignment")
    def recursive_kwargs_value_assignment(self, info, key_name, name_number):
        kwargs_value_assignment = ""
        if isinstance(info, dict):
            if info.get("type") == "torch.device" or info.get("type") == "torch.dtype":
                kwargs_value_assignment += "    " + "kwarg_" + key_name + name_number + " = " + info.get("value")
            else:
                kwargs_value_assignment += "    " + "kwarg_info_" + key_name + name_number + " = " + \
                    "{}".format(str(info)) + MsgConst.SPECIAL_CHAR[0]
                kwargs_value_assignment += "    " + "kwarg_" + key_name + name_number + " = " + \
                    "generate_data(kwarg_info_" + key_name + name_number + ")" + MsgConst.SPECIAL_CHAR[0]
            info["parameter_name"] = "kwarg_" + key_name + name_number
        else:
            for index, arg in enumerate(info):
                new_kwargs_value_assignment = self.recursive_kwargs_value_assignment(arg, key_name, name_number + \
                    "_" + str(index))
                kwargs_value_assignment += new_kwargs_value_assignment
        return kwargs_value_assignment

    def generate_kwargs_value_assignment_code(self, kwargs_info):
        kwargs_value_assignment = ""
        for key, value in kwargs_info.items():
            kwargs_value_assignment += self.recursive_kwargs_value_assignment(value, key, "")
        return kwargs_value_assignment

    @recursion_depth_decorator("OpGenerator: OperatorScriptGenerator.recursive_kwargs_dict")
    def recursive_kwargs_dict(self, info, flag_device=False, flag_bench=False):
        kwargs_dict_generator = ""
        if isinstance(info, dict):
            kwargs_dict_generator += info.get("parameter_name")
            if info.get("type") in TENSOR_DATA_LIST:
                if flag_device:
                    kwargs_dict_generator += ".to(device)"
                if flag_bench:
                    kwargs_dict_generator += '.to(torch.device("cpu"))'
                    kwargs_dict_generator += ".to(RAISE_PRECISION.get(str(" + info.get("parameter_name") + \
                        ".dtype), " + info.get("parameter_name") + ".dtype))"
        else:
            (left_bracket, right_bracket) = ("[", "]") if isinstance(info, list) else ("(", ")")
            kwargs_dict_generator += left_bracket
            for arg in info:
                kwargs_dict_generator += self.recursive_kwargs_dict(arg, flag_device=flag_device, flag_bench=flag_bench)
                kwargs_dict_generator += Const.COMMA
            kwargs_dict_generator += right_bracket
        return kwargs_dict_generator


    def generate_kwargs_dict(self, kwargs_info, flag_device):
        kwargs_dict_generator = ""
        for key, value in kwargs_info.items():
            kwargs_dict_generator += '"' + key + '"' + MonitorConst.NAME_SEP
            if flag_device:
                kwargs_dict_generator += self.recursive_kwargs_dict(value, flag_device=True) + Const.COMMA
            else:
                kwargs_dict_generator += self.recursive_kwargs_dict(value, flag_bench=True) + Const.COMMA
        return kwargs_dict_generator



def _op_generator_parser(parser):
    parser.add_argument("-i", "--config_input", dest="config_input", default='', type=str,
                        help="<Optional> Path of config json file", required=True)
    parser.add_argument("-o", "--api_output_path", dest="api_output_path", type=str,
                        help="<Required> Path of extract api_name.json.",
                        required=True)


def parse_json_config(json_file_path):
    if not json_file_path:
        config_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        json_file_path = os.path.join(config_dir, "config.json")
    json_config = load_json(json_file_path)
    common_config = CommonConfig(json_config)
    return common_config


def _run_operator_generate_commond(cmd_args):
    common_config = parse_json_config(cmd_args.config_input)

    if common_config.dump_json_path:
        api_extract = APIExtractor(common_config.api_name, common_config.dump_json_path, common_config.extract_api_path)
        api_extract.extract_op()
    check_file_or_directory_path(common_config.extract_api_path)
    check_file_or_directory_path(cmd_args.api_output_path, isdir=True)
    json_content = common_config.check_user_settings()
    api_info = APIInfo.from_json(json_content, common_config.propagation)

    if common_config.propagation == Const.BACKWARD:
        # read and check json
        api_full_name_forward, api_info_dict_forward = api_info.api_full_name, api_info.api_info_dict
        api_full_name_backward, api_info_dict_backward = (api_info.backward_info.api_full_name,
                                                          api_info.backward_info.api_info_dict)
        args_info_forward = api_info_dict_forward.get(Const.INPUT_ARGS)
        kwargs_info_forward = api_info_dict_forward.get(Const.INPUT_KWARGS)
        if Const.GRAD_INPUT in api_info_dict_backward:
            args_info_backward = api_info_dict_backward.get(Const.GRAD_INPUT)
        elif Const.INPUT in api_info_dict_backward:
            args_info_backward = api_info_dict_backward.get(Const.INPUT)
        op_generate = OperatorScriptGenerator(common_config, args_info_forward, kwargs_info_forward, args_info_backward)
        internal_settings = op_generate.get_settings(api_full_name_backward)
    else:
        # read and check json
        api_full_name_forward, api_info_dict_forward = api_info.api_full_name, api_info.api_info_dict
        args_info_forward = api_info_dict_forward.get(Const.INPUT_ARGS)
        kwargs_info_forward = api_info_dict_forward.get(Const.INPUT_KWARGS)
        op_generate = OperatorScriptGenerator(common_config, args_info_forward, kwargs_info_forward, None)
        internal_settings = op_generate.get_settings(api_full_name_forward)

    template_path = os.path.join(os.path.dirname(__file__), "operator_replication.template")
    operator_script_path = os.path.join(cmd_args.api_output_path, 
                                        "{0}.py".format(internal_settings.get("api_full_name")))

    try:
        with FileOpen(template_path, 'r') as ftemp, FileOpen(operator_script_path, 'w') as fout:
            code_template = ftemp.read()
            fout.write(code_template.format(**internal_settings))
    except OSError:
        logger.error(f"Failed to open file. Please check file {template_path} or {operator_script_path}.")

    logger.info(f"Generate operator script successfully and the name is {operator_script_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _op_generator_parser(parser)
    cmd_args = parser.parse_args()
    _run_operator_generate_commond(cmd_args)
