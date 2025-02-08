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

import re
import abc
from mindspore import Tensor

from msprobe.core.common.log import logger


# 用于存储所有validator实现类的注册表
config_validator_registry = {}


def register_config_validator(cls):
    """装饰器 用于注册ConfigValidator的实现类"""
    config_validator_registry[cls.__name__] = cls
    return cls


class ConfigValidator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def check_pattern_match(self, config_spec: str):
        pass

    @abc.abstractmethod
    def validate(self, actual_data, module_name: str, data_type: str, pattern_match):
        pass


@register_config_validator
class TensorValidator(ConfigValidator):
    def check_pattern_match(self, config_spec: str):
        pattern = re.compile(r"tensor")
        return pattern.match(config_spec)

    def validate(self, actual_data, module_name: str, data_type: str, pattern_match):
        if not isinstance(actual_data, Tensor):
            raise ValueError(
                f"Format of {module_name} {data_type} does not match the required format 'tensor' in config.")


@register_config_validator
class TupleValidator(ConfigValidator):
    def check_pattern_match(self, config_spec: str):
        pattern = re.compile(r"tuple\[(\d+)\]:?(\d+)?")
        return pattern.match(config_spec)

    def validate(self, actual_data, module_name: str, data_type: str, pattern_match):
        length, index = pattern_match.groups()
        if index is None:
            index = 0
        length, index = int(length), int(index)

        if not (0 <= index < length):
            raise ValueError(
                f"Format of {module_name} {data_type} in config.json does not match the required format 'tuple[x]:y'."
                f"y must be greater than or equal to 0 and less than x.")
        if not isinstance(actual_data, tuple):
            raise ValueError(
                f"Type of {module_name} {data_type} does not match spec of config.json, should be tuple, please check.")
        if len(actual_data) != length:
            raise ValueError(
                f"Length of {module_name} {data_type} does not match spec of config.json, should be {length}, "
                f"actual is {len(actual_data)} please check.")
        return index


def validate_config_spec(config_spec: str, actual_data, module_name: str, data_type: str):
    focused_col = None
    for _, validator_cls in config_validator_registry.items():
        config_validator = validator_cls()
        pattern_match = config_validator.check_pattern_match(config_spec)
        if pattern_match:
            try:
                focused_col = config_validator.validate(actual_data, module_name, data_type, pattern_match)
            except ValueError as e:
                logger.warning(f"config spec validate failed: {str(e)}")
            return focused_col
    logger.warning(f"config spec in {module_name} {data_type} not supported, "
                   f"expected spec:'tuple\[(\d+)\]:(\d+)' or 'tensor', actual spec: {config_spec}.")
    return focused_col