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
    def _load_cases(cls, filename):
        """从JSON文件加载测试用例"""
        path = os.path.join(cls.CASE_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
