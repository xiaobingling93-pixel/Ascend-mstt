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
import json
import os


class TestCaseFactory:
    """管理所有测试用例的统一工厂"""
    
    CASE_DIR = os.path.join(os.path.dirname(__file__), 'ut_test_cases')

    @classmethod
    def get_process_task_add_cases(cls):
        return cls._load_cases('test_match_node_controller\\process_task_add_case.json')
    
    @classmethod
    def get_process_task_delete_cases(cls):
        return cls._load_cases('test_match_node_controller\\process_task_delete_case.json')
    
    @classmethod
    def get_process_task_add_child_layer_cases(cls):
        return cls._load_cases('test_match_node_controller\\process_task_add_child_layer.json')
    
    @classmethod
    def get_process_task_delete_child_layer_cases(cls):
        return cls._load_cases('test_match_node_controller\\process_task_delete_child_layer.json')
    
    @classmethod
    def get_process_task_add_child_layer_by_config_cases(cls):
        return cls._load_cases('test_match_node_controller\\process_task_add_child_layer_by_config.json')
    
    @classmethod
    def _load_cases(cls, filename):
        """从JSON文件加载测试用例"""
        path = os.path.join(cls.CASE_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
