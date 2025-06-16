import pytest

from server.app.controllers.match_nodes_controller import MatchNodesController
from tests.data.test_case_factory import TestCaseFactory


@pytest.mark.unit
class TestMatchNodesController:
    """测试匹配节点功能"""

    @pytest.mark.parametrize("test_case", TestCaseFactory.get_process_task_add_cases(),
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_process_task_add(self, test_case):
        """测试添加子节点层功能"""
        graph_data, npu_node_name, bench_node_name, task = test_case['input'].values()
        expected = test_case['expected']
        actual = MatchNodesController.process_task_add(graph_data, npu_node_name, bench_node_name, task)
        assert actual == expected
        
    @pytest.mark.parametrize("test_case", TestCaseFactory.get_process_task_delete_cases(),
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_process_task_delete(self, test_case):
        """测试删除子节点层功能"""
        graph_data, npu_node_name, bench_node_name, task = test_case['input'].values()
        expected = test_case['expected']
        actual = MatchNodesController.process_task_delete(graph_data, npu_node_name, bench_node_name, task)
        assert actual == expected
   
