import json
import unittest
from unittest.mock import patch, MagicMock
from msprobe.visualization.compare.mode_adapter import ModeAdapter
from msprobe.visualization.graph.base_node import BaseNode, NodeOp
from msprobe.visualization.utils import GraphConst, ToolTip
from msprobe.core.common.const import CompareConst


class TestModeAdapter(unittest.TestCase):

    def setUp(self):
        self.node_op = NodeOp.module
        self.node_id = "node_1"
        self.node = BaseNode(self.node_op, self.node_id)
        self.compare_mode = GraphConst.REAL_DATA_COMPARE
        self.adapter = ModeAdapter(self.compare_mode)
        self.compare_data_dict = [{}, {}]

    def test_add_md5_compare_data(self):
        node_data = {'md5_key': 'some_md5_value'}
        compare_data_dict = {'md5_key': 'expected_md5_value'}
        precision_index = ModeAdapter._add_md5_compare_data(node_data, compare_data_dict)
        self.assertEqual(precision_index, 1)

    @patch('msprobe.visualization.compare.mode_adapter.ModeAdapter')
    def test_parse_result(self, mock_mode_adapter):
        mock_mode_adapter._add_summary_compare_data.return_value = 0.5
        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        precision_index, other_dict = self.adapter.parse_result(self.node, self.compare_data_dict)
        self.assertEqual(precision_index, 0.5)
        self.assertEqual(other_dict, {})

        mock_mode_adapter._add_md5_compare_data.return_value = 1
        self.adapter.compare_mode = GraphConst.MD5_COMPARE
        precision_index, other_dict = self.adapter.parse_result(self.node, self.compare_data_dict)
        self.assertEqual(precision_index, 1)
        self.assertEqual(other_dict, {'Result': 'pass'})

        mock_mode_adapter._add_real_compare_data.return_value = 0.6
        self.adapter.compare_mode = GraphConst.REAL_DATA_COMPARE
        precision_index, other_dict = self.adapter.parse_result(self.node, self.compare_data_dict)
        self.assertEqual(precision_index, 0.0)
        self.assertEqual(other_dict, {})

    def test_prepare_real_data(self):
        result = self.adapter.prepare_real_data(self.node)
        self.assertTrue(result)

        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        result = self.adapter.prepare_real_data(self.node)
        self.assertFalse(result)

    def test_add_csv_data(self):
        compare_result_list = ['result1', 'result2']
        self.adapter.add_csv_data(compare_result_list)
        self.assertEqual(self.adapter.csv_data, compare_result_list)

    def test_add_error_key(self):
        node_data = {'key': {}}
        self.adapter.compare_mode = GraphConst.REAL_DATA_COMPARE
        self.adapter.add_error_key(node_data)
        self.assertEqual(node_data['key'][GraphConst.ERROR_KEY],
                         [CompareConst.ONE_THOUSANDTH_ERR_RATIO, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO])
        node_data = {'key': {}}
        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        self.adapter.add_error_key(node_data)
        self.assertEqual(node_data['key'][GraphConst.ERROR_KEY],
                         [CompareConst.MAX_RELATIVE_ERR, CompareConst.MIN_RELATIVE_ERR,
                          CompareConst.MEAN_RELATIVE_ERR, CompareConst.NORM_RELATIVE_ERR])

    def test_get_tool_tip(self):
        self.adapter.compare_mode = GraphConst.MD5_COMPARE
        tips = self.adapter.get_tool_tip()
        self.assertEqual(tips, json.dumps({'md5': ToolTip.MD5}))

        self.adapter.compare_mode = GraphConst.SUMMARY_COMPARE
        tips = self.adapter.get_tool_tip()
        self.assertEqual(tips, json.dumps({
            CompareConst.MAX_DIFF: ToolTip.MAX_DIFF,
            CompareConst.MIN_DIFF: ToolTip.MIN_DIFF,
            CompareConst.MEAN_DIFF: ToolTip.MEAN_DIFF,
            CompareConst.NORM_DIFF: ToolTip.NORM_DIFF}))

        self.adapter.compare_mode = GraphConst.REAL_DATA_COMPARE
        tips = self.adapter.get_tool_tip()
        self.assertEqual(tips, json.dumps({
            CompareConst.ONE_THOUSANDTH_ERR_RATIO: ToolTip.ONE_THOUSANDTH_ERR_RATIO,
            CompareConst.FIVE_THOUSANDTHS_ERR_RATIO: ToolTip.FIVE_THOUSANDTHS_ERR_RATIO,
            CompareConst.COSINE: ToolTip.COSINE,
            CompareConst.MAX_ABS_ERR: ToolTip.MAX_ABS_ERR,
            CompareConst.MAX_RELATIVE_ERR: ToolTip.MAX_RELATIVE_ERR}))
