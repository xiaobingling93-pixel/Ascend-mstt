# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import re

# 模块级全局变量

ADD_MATCH_KEYS = [
    'MaxAbsErr',
    'MinAbsErr',
    'MeanAbsErr',
    'NormAbsErr',
    'MaxRelativeErr',
    'MinRelativeErr',
    'MeanRelativeErr',
    'NormRelativeErr',
]
FILE_NAME_REGEX = r'^[a-zA-Z0-9_\-\.]+$'  # 文件名正则表达式

_state = {'logdir': '', "current_tag": '', "current_file_path": '', 'current_file_data': {}}


def init_defaults():
    """
    初始化全局变量的默认值
    """
    global _state
    _state = {'logdir': '', "current_tag": '', "current_file_path": '', 'current_file_data': {}}


def set_global_value(key, value):
    """
    设置全局变量的值
    """
    _state[key] = value


def get_global_value(key, default=None):
    """
    获取全局变量的值
    """
    return _state.get(key, default)


def reset_global_state():
    """
    重置所有全局变量为默认值
    """
    init_defaults()
