#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

# 标准库
import argparse
import json
import os
import re
import string

# 应用程序自定义模块
from msprobe.core.common.file_utils import (
    FileOpen,
    load_json,
    save_json,
    make_dir,
    change_mode,
)
from msprobe.core.common.utils import (
    check_file_or_directory_path,
    check_op_str_pattern_valid,
    is_int,
)
from msprobe.core.common.const import Const, MonitorConst, MsgConst, FileCheckConst
from msprobe.core.common.log import logger
from msprobe.core.common.decorator import recursion_depth_decorator

OPERATOR_TYPE = ("Functional", "Tensor", "Torch", "Mint")

API_INFO = 2
FOUR_SEGMENT = 4
FIVE_SEGMENT = 5
DATA_NAME = "data_name"
API_MAX_LENGTH = 300
PROPAGATION_LIST = [Const.FORWARD, Const.BACKWARD]
DATAMODE_LIST = ["random_data", "real_data"]
ITER_MAX_TIMES = 1000
FRAMEWORK = 'framework'
REAL_DATA_PATH = 'real_data_path'
EXCLUED = {FRAMEWORK, REAL_DATA_PATH}


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
        if iter_t <= 0 or iter_t > ITER_MAX_TIMES:
            raise ValueError(f"iter_times should be range from 1 to {ITER_MAX_TIMES}.")

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

        filtered = {k: v for k, v in json_content.items() if k not in EXCLUED}

        if not filtered:
            raise ValueError(f'json file is empty!')

        if len(filtered) > API_INFO:
            raise ValueError(f'json file has more than one API, the API only contains forward and backward info')

        is_forward_phase = propagation == Const.FORWARD

        is_exact_api_count = len(filtered) == API_INFO

        all_keys_forward = all(k.endswith('forward') for k in filtered)

        if is_forward_phase and is_exact_api_count and all_keys_forward:
            raise ValueError(
                "json file has more than one API, the API only contains forward info。"
            )

        # Retrieve the first API name and dictionary
        forward_item = next(iter(json_content.items()), None)
        if not forward_item or not isinstance(forward_item[1], dict) or not forward_item[1]:
            raise ValueError(f'Invalid forward API data in json_content!')

        # if propagation is backward, ensure json file contains forward and backward info
        if propagation == Const.BACKWARD and len(filtered) < API_INFO:
            raise ValueError(f'Backward propagation requires contains forward and backward info!')

        # if propagation is backward, ensure it has valid data
        if propagation == Const.BACKWARD:
            backward_item = list(json_content.items())[1]
            if not isinstance(backward_item[1], dict) or not backward_item[1]:
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
        self.framework = None
        self.real_data_path = None

    def extract_op(self):
        self.data = load_json(self.dump_json_path)
        # 拿到 framework
        self.framework = self.data.get(FRAMEWORK, None)

        new_data = {}
        extract_key_pattern = re.compile(f"^{re.escape(self.api_name)}\..+")  # 修改为只要包含或等于apiname即可，不需要是只包含

        self.real_data_path = self.data.get('dump_data_dir', '')

        for key, value in self.data.get('data', {}).items():
            if extract_key_pattern.match(key):
                if self.real_data_path:
                    value = self.load_real_data_path(value, self.real_data_path)
                new_data[key] = value

        if self.real_data_path is not None:
            new_data[REAL_DATA_PATH] = self.real_data_path

        # 把 framework 加进去
        if self.framework is not None:
            new_data[FRAMEWORK] = self.framework
        if not new_data:
            logger.warning(f"Warning: The api '{self.api_name}' does not exist in the file.")
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

    @recursion_depth_decorator("OpGenerator: APIExtractor.update_data_name")
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

    @staticmethod
    def generate_forward_inputs_code(args_info):
        names = []

        def collect(info):
            if isinstance(info, dict):
                names.append(info["parameter_name"])
            else:
                for sub in info:
                    collect(sub)

        collect(args_info)

        return (
                "    forward_inputs = [\n"
                "        ComputeElement(parameter=info)\n"
                "        for info in (" + ", ".join(names) + ")\n"
                                                             "    ]\n"
        )

    @staticmethod
    def generate_kwargs_compute_element_dict_code():
        return (
            "    # ---- 构造 kwargs 对应的 ComputeElement 字典 ----\n"
            "    kwargs_compute_element_dict = {\n"
            "        key_str: ComputeElement(compute_element_info=compute_element_info)\n"
            "        for key_str, compute_element_info in kwargs_device.items()\n"
            "    }\n"
        )

    @staticmethod
    def generate_gradient_inputs_code(args_info_backward):
        names = []

        def collect(info):
            if isinstance(info, dict):
                names.append(info["parameter_name"])
            else:
                for sub in info:
                    collect(sub)

        collect(args_info_backward)

        return (
                "    # —— 构造反向梯度 ComputeElement 列表 —— #\n"
                "    gradient_inputs = [\n"
                "        ComputeElement(parameter=info)\n"
                "        for info in (" + ", ".join(names) + ")\n"
                                                             "    ]\n"
        )

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
        internal_settings["ordinal_number"] = ordinal_number
        internal_settings["direction_status"] = self.common_config.propagation
        internal_settings["random_seed"] = self.common_config.random_seed
        internal_settings["data_mode"] = self.common_config.data_mode
        if self.common_config.data_mode == "real_data":
            internal_settings["iter_times"] = 1
        else:
            internal_settings["iter_times"] = self.common_config.iter_times

        internal_settings["args_info_forward"] = self.args_info_forward
        internal_settings["kwargs_info_forward"] = self.kwargs_info_forward
        internal_settings["args_info_backward"] = self.args_info_backward

        return internal_settings


def _op_generator_parser(parser):
    parser.add_argument("-i", "--config_input", dest="config_input", type=str,
                        help="<Required> Path of config json file", required=True)
    parser.add_argument("-o", "--api_output_path", dest="api_output_path", type=str,
                        help="<Required> Path of extract api_name.json.", required=True)


def parse_json_config(json_file_path):
    if not json_file_path:
        raise Exception("config_input path can not be empty, please check.")
    json_config = load_json(json_file_path)
    common_config = CommonConfig(json_config)
    return common_config


def _run_operator_generate_commond(cmd_args):
    common_config = parse_json_config(cmd_args.config_input)

    if common_config.dump_json_path:
        api_extract = APIExtractor(common_config.api_name, common_config.dump_json_path, common_config.extract_api_path)
        api_extract.extract_op()
        framework = api_extract.framework
        real_data_path = api_extract.real_data_path
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
        internal_settings[FRAMEWORK] = framework
        internal_settings[REAL_DATA_PATH] = real_data_path
    else:
        # read and check json
        api_full_name_forward, api_info_dict_forward = api_info.api_full_name, api_info.api_info_dict

        args_info_forward = api_info_dict_forward.get(Const.INPUT_ARGS)

        kwargs_info_forward = api_info_dict_forward.get(Const.INPUT_KWARGS)

        op_generate = OperatorScriptGenerator(common_config, args_info_forward, kwargs_info_forward, None)
        internal_settings = op_generate.get_settings(api_full_name_forward)
        internal_settings[FRAMEWORK] = framework
        internal_settings[REAL_DATA_PATH] = real_data_path

    template_path = os.path.join(os.path.dirname(__file__), "operator_replication.template")
    operator_script_path = os.path.join(cmd_args.api_output_path,
                                        "{0}.py".format(internal_settings.get("api_full_name")))

    class SafeDict(dict):
        def __missing__(self, key):
            # leave {key} in the output if it’s not in the dict
            return '{' + key + '}'

    class RobustFormatter(string.Formatter):
        def vformat(self, format_string, args, kwargs):
            result = []
            # parse() 会把文本和每个占位符拆开
            for literal, field_name, format_spec, conversion in self.parse(format_string):
                # 输出字面文本
                result.append(literal)
                if field_name is None:
                    continue
                try:
                    # 正常获取变量并格式化
                    obj, _ = self.get_field(field_name, args, kwargs)
                    if conversion:
                        obj = self.convert_field(obj, conversion)
                    result.append(self.format_field(obj, format_spec))
                except Exception:
                    # 不管是 KeyError 还是 ValueError，都原样回写 {field_name[:format_spec]}
                    placeholder = '{' + field_name
                    if conversion:
                        placeholder += '!' + conversion
                    if format_spec:
                        placeholder += ':' + format_spec
                    placeholder += '}'
                    result.append(placeholder)
            return ''.join(result)

    fmt = RobustFormatter()
    with FileOpen(template_path, 'r') as ftemp, FileOpen(operator_script_path, 'w') as fout:
        code_template = ftemp.read()
        # 这里用 fmt.format，不用 format_map
        fout.write(fmt.format(code_template, **internal_settings))

    change_mode(operator_script_path, FileCheckConst.DATA_FILE_AUTHORITY)

    logger.info(f"Generate operator script successfully and the name is {operator_script_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _op_generator_parser(parser)
    cmd_args = parser.parse_args()
    _run_operator_generate_commond(cmd_args)
