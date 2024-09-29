#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import os
from msprobe.core.common.file_utils import load_yaml, check_file_or_directory_path
from msprobe.pytorch.pt_config import RunUTConfig


class Config:
    def __init__(self, yaml_file):
        check_file_or_directory_path(yaml_file, False)
        config = load_yaml(yaml_file)
        self.config = {key: self.validate(key, value) for key, value in config.items()}

    def __getattr__(self, item):
        return self.config[item]

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    @staticmethod
    def validate(key, value):
        validators = {
            'white_list': list,
            'black_list': list,
            'error_data_path': str,
            'precision': int,
            'is_online': bool,
            'nfs_path': str,
            'host': str,
            'port': int,
            'rank_list': list,
            'tls_path': str
        }
        if key not in validators:
            raise ValueError(f"{key} must be one of {validators.keys()}")
        if not isinstance(value, validators.get(key)):
            raise ValueError(f"{key} must be {validators[key].__name__} type")
        if key == 'precision' and (value < 0 or value > 20):
            raise ValueError("precision must be greater than or equal to 0 and less than 21")
        if key == 'white_list':
            RunUTConfig.check_filter_list_config(key, value)
        if key == 'black_list':
            RunUTConfig.check_filter_list_config(key, value)
        if key == 'error_data_path':
            RunUTConfig.check_error_data_path_config(value)
        if key == 'nfs_path':
            RunUTConfig.check_nfs_path_config(value)
        if key == 'tls_path':
            RunUTConfig.check_tls_path_config(value)
        return value


cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)
