import pytest
from server.app.service.graph_service import GraphService
from server.app.utils.graph_utils import GraphUtils


class TestMatchNodesService:

    @pytest.fixture(autouse=True)
    def setup_manager(self, meta_data):
        self.graph_data = GraphUtils.get_graph_data(meta_data)
        
    def test_service(self, meta_data, operation_config):
        op_type = operation_config["type"]
        expected = operation_config["expected"]
         # 执行操作
        try:
            if op_type == "get_node_info":
                result = GraphService.get_node_info(
                    node_info=operation_config["node_info"],
                    meta_data=meta_data
                )
            elif op_type == "add_match_nodes":
                result = GraphService.add_match_nodes(
                    npu_node_name=operation_config["npu_node_name"],
                    bench_node_name=operation_config["bench_node_name"],
                    meta_data=meta_data
                )
            elif op_type == "delete_match_nodes":
                result = GraphService.delete_match_nodes(
                    npu_node_name=operation_config["npu_node_name"],
                    bench_node_name=operation_config["bench_node_name"],
                    meta_data=meta_data
                )
            elif op_type == "get_matched_state_list":
                result = GraphService.get_matched_state_list(
                    meta_data=meta_data
                )
            elif op_type == "save_data":
                result = GraphService.save_data(
                    meta_data=meta_data
                )
        except Exception as e:
            result = {"error": type(e).__name__}
        
        # 验证结果
        assert result == expected, \
            f"Operation {op_type} failed on {operation_config}"
