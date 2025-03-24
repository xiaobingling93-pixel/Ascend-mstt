import pytest
from server.app.controllers.match_nodes_controller import MatchNodesController
from server.app.utils.graph_utils import GraphUtils


class TestMatchNodesController:

    @pytest.fixture(autouse=True)
    def setup_manager(self, graph_data):
        self.graph_data = GraphUtils.get_graph_data(graph_data)

