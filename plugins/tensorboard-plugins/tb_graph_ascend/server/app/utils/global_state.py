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

# 全局常量
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
MAX_FILE_SIZE = 3 * 1024 * 1024 * 1024  # 最大文件大小限制
NPU_PREFIX = 'N___'
BENCH_PREFIX = 'B___'
FILE_NAME_REGEX = r'^[a-zA-Z0-9_\-\.]+$'  # 文件名正则表达式
# 图类型
NPU = 'NPU'
BENCH = 'Bench'
SINGLE = 'Single'
# 前端节点类型
EXPAND_MODULE = 0
UNEXPAND_NODE = 1
# 权限码
PERM_GROUP_WRITE = 0o020
PERM_OTHER_WRITE = 0o002

# 后端节点类型
MODULE = 0
API = 1
MULTI_COLLECTION = 8
API_LIST = 9


class GraphState:
    # 模块级全局变量
    _state = {
        'logdir': '',
        'current_tag': '',
        'current_run': '',
        'current_file_path': '',
        'current_file_data': {},
        'current_hierarchy': {},
        'config_data': {
            "run": "",
            "npuUnMatchNodes": [],
            "benchUnMatchNodes": [],
            "npuMatchNodes": {},
            "benchMatchNodes": {},
            "manualMatchNodes": {},
            
        },
        'first_run_tags': {},
        'runs': {},
    }

    @staticmethod
    def init_defaults():
        """
        初始化全局变量的默认值
        """
        global _state
        GraphState._state = {
            'logdir': '',
            'current_tag': '',
            'current_run': '',
            'current_file_path': '',
            'current_file_data': {},
            'current_hierarchy': {},
            'config_data': {
                "run": "",
                "npuUnMatchNodes": [],
                "benchUnMatchNodes": [],
                "npuMatchNodes": {},
                "benchMatchNodes": {},
                "manualMatchNodes": {},
                
            },
            'first_run_tags': {},
            'runs': {},
        }

    @staticmethod
    def set_global_value(key, value):
        """
        设置全局变量的值
        """
        GraphState._state[key] = value

    @staticmethod
    def get_global_value(key, default=None):
        """
        获取全局变量的值
        """
        return GraphState._state.get(key, default)

    @staticmethod
    def reset_global_state():
        """
        重置所有全局变量为默认值
        """
        GraphState.init_defaults()
