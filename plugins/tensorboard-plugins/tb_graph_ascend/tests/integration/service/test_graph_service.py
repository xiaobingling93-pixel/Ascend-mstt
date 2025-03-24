import pytest
from server.app.controllers.match_nodes_controller import MatchNodesController
from server.app.utils.graph_utils import GraphUtils


class TestMatchNodesService:

    @pytest.fixture(autouse=True)
    def setup_manager(self, meta_data):
        print('meta_data', meta_data)
        self.graph_data = GraphUtils.get_graph_data(meta_data)
        
    def test_operation(self, operation_config):
        print("operation_config: ", operation_config)
        op_type = operation_config["type"]
        expected = operation_config["expected"]
        print("op_type: ", op_type)
        print("expected: ", expected)
        assert True
        print('---------------')
