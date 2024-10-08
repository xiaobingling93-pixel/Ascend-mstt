import unittest
from unittest.mock import MagicMock, patch
from msprobe.core.data_dump.scope import ModuleRangeScope
from msprobe.core.common.const import Const
from msprobe.mindspore.cell_processor import CellProcessor  # 替换为实际的模块名


class MockCell:
    def __init__(self):
        self.mindstudio_reserved_name = None


class TestCellProcessor(unittest.TestCase):

    def setUp(self):
        # 重置静态变量
        CellProcessor.reset_cell_stats()
        self.scope = MagicMock(spec=ModuleRangeScope)
        self.processor = CellProcessor(self.scope)

    def test_init_with_module_range_scope(self):
        self.assertIsInstance(self.processor.scope, ModuleRangeScope)

    def test_init_with_none_scope(self):
        mock_const.SEP = "."  # 确保 SEPARATOR 设置为字符串
        processor = CellProcessor(None)
        self.assertIsNone(processor.scope)

    def test_set_cell_count_new_cell(self):
        count = self.processor.set_cell_count("cell1")
        self.assertEqual(count, 0)
        self.assertEqual(CellProcessor.cell_count["cell1"], 0)

    def test_set_cell_count_existing_cell(self):
        self.processor.set_cell_count("cell1")
        count = self.processor.set_cell_count("cell1")
        self.assertEqual(count, 1)
        self.assertEqual(CellProcessor.cell_count["cell1"], 1)

    def test_reset_cell_stats(self):
        self.processor.set_cell_count("cell1")
        CellProcessor.reset_cell_stats()
        self.assertEqual(CellProcessor.cell_count, {})
        self.assertEqual(CellProcessor.cell_stack, [])
        self.assertEqual(CellProcessor.api_parent_node, "")
        self.assertEqual(CellProcessor.module_node, {})

    @patch('msprobe.core.common.const.Const')
    def test_node_hook_begin(self, mock_const):
        mock_const.SEP = "."  # 确保 SEPARATOR 设置为字符串
        mock_const.START = "start"
        cell = MockCell()
        self.processor.node_hook("prefix", "start")(cell, "input")

        expected_name = "prefix" + mock_const.SEP + "0"
        self.assertEqual(cell.mindstudio_reserved_name, expected_name)
        self.assertIn(expected_name, CellProcessor.cell_stack)
        self.assertEqual(CellProcessor.api_parent_node, expected_name)
        self.scope.begin_module.assert_called_once_with(expected_name)

    @patch('msprobe.core.common.const.Const')
    def test_node_hook_end(self, mock_const):
        mock_const.START = "start"
        cell = MockCell()
        self.processor.node_hook("prefix", "start")(cell, "input")
        self.processor.node_hook("prefix", "stop")(cell, "input", "output")

        self.assertEqual(len(CellProcessor.cell_stack), 0)
        self.assertIsNone(CellProcessor.api_parent_node)
        self.scope.end_module.assert_called_once_with(cell.mindstudio_reserved_name)

    @patch('msprobe.core.common.const.Const')
    def test_multiple_node_hook_calls(self, mock_const):
        mock_const.START = "start"
        cell = MockCell()

        # First call
        self.processor.node_hook("prefix", "start")(cell, "input")
        expected_name1 = "prefix" + mock_const.SEP + "0"

        # Second call
        self.processor.node_hook("prefix", "start")(cell, "input")
        expected_name2 = "prefix" + mock_const.SEP + "1"

        self.assertEqual(cell.mindstudio_reserved_name, expected_name2)
        self.assertEqual(CellProcessor.api_parent_node, expected_name2)

        # End first call
        self.processor.node_hook("prefix", "stop")(cell, "input", "output")
        self.assertEqual(len(CellProcessor.cell_stack), 1)  # Still one item in stack
        self.assertEqual(CellProcessor.api_parent_node, expected_name1)

        # End second call
        self.processor.node_hook("prefix", "stop")(cell, "input", "output")
        self.assertEqual(len(CellProcessor.cell_stack), 0)  # Stack should be empty now
        self.assertIsNone(CellProcessor.api_parent_node)


if __name__ == "__main__":
    unittest.main()
