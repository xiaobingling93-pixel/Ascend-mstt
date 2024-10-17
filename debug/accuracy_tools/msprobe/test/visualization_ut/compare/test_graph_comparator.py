import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from msprobe.visualization.compare.graph_comparator import GraphComparator
from msprobe.visualization.graph.graph import Graph, BaseNode, NodeOp
from msprobe.visualization.utils import GraphConst


class TestGraphComparator(unittest.TestCase):

    def setUp(self):
        self.graphs = [Graph("model1"), Graph("model2")]
        self.data_paths = ["step1/rank/dump.json", "step2/rank/dump.json"]
        self.stack_path = "step1/rank/stack.json"
        self.output_path = "output/output.vis"

    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    @patch('msprobe.visualization.compare.graph_comparator.load_json_file')
    @patch('msprobe.visualization.compare.graph_comparator.load_data_json_file')
    def test__parse_param(self, mock_load_data_json_file, mock_load_json_file, mock_get_compare_mode):
        mock_load_data_json_file.return_value = "data_dict"
        mock_load_json_file.return_value = "construct_dict"
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        self.comparator = GraphComparator(self.graphs, self.data_paths, self.stack_path, self.output_path)
        self.comparator._parse_param(self.data_paths, self.stack_path, self.output_path)

        self.assertEqual(self.comparator.dump_path_param, {
            'npu_json_path': self.data_paths[0],
            'bench_json_path': self.data_paths[1],
            'stack_json_path': self.stack_path,
            'is_print_compare_log': True
        })
        self.assertEqual(self.comparator.output_path, self.output_path)

    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    @patch('msprobe.visualization.compare.graph_comparator.load_json_file')
    @patch('msprobe.visualization.compare.graph_comparator.load_data_json_file')
    def test_compare(self, mock_load_data_json_file, mock_load_json_file, mock_get_compare_mode):
        mock_load_data_json_file.return_value = "data_dict"
        mock_load_json_file.return_value = "construct_dict"
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        comparator = GraphComparator(self.graphs, self.data_paths, self.stack_path, self.output_path)
        comparator._compare_nodes = MagicMock()
        comparator._postcompare = MagicMock()

        comparator.compare()

        comparator._compare_nodes.assert_called_once()
        comparator._postcompare.assert_called_once()

    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    @patch('msprobe.visualization.compare.graph_comparator.load_json_file')
    @patch('msprobe.visualization.compare.graph_comparator.load_data_json_file')
    def test_add_compare_result_to_node(self, mock_load_data_json_file, mock_load_json_file, mock_get_compare_mode):
        mock_load_data_json_file.return_value = "data_dict"
        mock_load_json_file.return_value = "construct_dict"
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        node = MagicMock()
        compare_result_list = [("output1", "data1"), ("input1", "data2")]

        comparator = GraphComparator(self.graphs, self.data_paths, self.stack_path, self.output_path)
        comparator.ma = MagicMock()
        comparator.ma.prepare_real_data.return_value = True

        comparator.add_compare_result_to_node(node, compare_result_list)
        comparator.ma.prepare_real_data.assert_called_once_with(node)
        node.data.update.assert_not_called()

    @patch('msprobe.visualization.graph.node_colors.NodeColors.get_node_error_status')
    @patch('msprobe.visualization.utils.get_csv_df')
    @patch('msprobe.visualization.builder.msprobe_adapter.run_real_data')
    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    @patch('msprobe.visualization.compare.graph_comparator.load_json_file')
    @patch('msprobe.visualization.compare.graph_comparator.load_data_json_file')
    def test__postcompare(self, mock_load_data_json_file, mock_load_json_file, mock_get_compare_mode,
                          mock_run_real_data, mock_get_csv_df, mock_get_node_error_status):
        mock_load_data_json_file.return_value = "data_dict"
        mock_load_json_file.return_value = "construct_dict"
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        mock_df = MagicMock()
        mock_df.iterrows = MagicMock(return_value=[(None, MagicMock())])
        mock_run_real_data.return_value = mock_df
        mock_get_csv_df.return_value = mock_df
        mock_get_node_error_status.return_value = True
        comparator = GraphComparator(self.graphs, self.data_paths, self.stack_path, self.output_path)
        comparator.ma = MagicMock()
        comparator.ma.is_real_data_compare.return_value = True
        comparator._handle_api_collection_index = MagicMock()
        comparator.ma.compare_nodes = [MagicMock()]
        comparator.ma.parse_result = MagicMock(return_value=(0.9, None))

        comparator._postcompare()

        comparator._handle_api_collection_index.assert_called_once()
        comparator.ma.add_error_key.assert_called()

    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    @patch('msprobe.visualization.compare.graph_comparator.load_json_file')
    @patch('msprobe.visualization.compare.graph_comparator.load_data_json_file')
    def test__handle_api_collection_index(self, mock_load_data_json_file, mock_load_json_file, mock_get_compare_mode):
        mock_load_data_json_file.return_value = "data_dict"
        mock_load_json_file.return_value = "construct_dict"
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        comparator = GraphComparator(self.graphs, self.data_paths, self.stack_path, self.output_path)
        apis = BaseNode(NodeOp.api_collection, 'Apis_Between_Modules.0')
        api1 = BaseNode(NodeOp.function_api, 'Tensor.a.0')
        api1.data = {GraphConst.JSON_INDEX_KEY: 0.9}
        api2 = BaseNode(NodeOp.function_api, 'Tensor.b.0')
        api2.data = {GraphConst.JSON_INDEX_KEY: 0.6}
        apis.subnodes = [api1, api2]
        sub_nodes = [BaseNode(NodeOp.module, 'Module.a.0'), apis, BaseNode(NodeOp.module, 'Module.a.1')]
        comparator.graph_n.root.subnodes = sub_nodes
        comparator._handle_api_collection_index()
        self.assertEqual(comparator.graph_n.root.subnodes[1].data.get(GraphConst.JSON_INDEX_KEY), 0.6)

    @patch('msprobe.visualization.builder.msprobe_adapter.compare_node')
    @patch('msprobe.visualization.graph.graph.Graph.match')
    @patch('msprobe.visualization.graph.graph.Graph.mapping_match')
    @patch('msprobe.visualization.compare.graph_comparator.get_compare_mode')
    @patch('msprobe.visualization.compare.graph_comparator.load_json_file')
    @patch('msprobe.visualization.compare.graph_comparator.load_data_json_file')
    def test__compare_nodes(self, mock_load_data_json_file, mock_load_json_file, mock_get_compare_mode,
                            mock_mapping_match, mock_match, mock_compare_node):
        node_n = BaseNode(NodeOp.function_api, 'Tensor.a.0')
        node_b = BaseNode(NodeOp.function_api, 'Tensor.b.0')
        mock_load_data_json_file.return_value = {}
        mock_load_json_file.return_value = {}
        mock_get_compare_mode.return_value = GraphConst.SUMMARY_COMPARE
        mock_mapping_match.return_value = (node_b, [], [])
        mock_compare_node.return_value = ['result']
        comparator = GraphComparator(self.graphs, self.data_paths, self.stack_path, self.output_path)
        comparator.mapping_config = True
        comparator._compare_nodes(node_n)
        self.assertEqual(node_n.matched_node_link, ['Tensor.b.0'])
        self.assertEqual(node_b.matched_node_link, ['Tensor.a.0'])
        comparator.mapping_config = False
        node_n = BaseNode(NodeOp.function_api, 'Tensor.a.0')
        node_b = BaseNode(NodeOp.function_api, 'Tensor.a.0')
        mock_match.return_value = (node_b, [])
        comparator._compare_nodes(node_n)
        self.assertEqual(node_n.matched_node_link, ['Tensor.a.0'])
        self.assertEqual(node_b.matched_node_link, ['Tensor.a.0'])

