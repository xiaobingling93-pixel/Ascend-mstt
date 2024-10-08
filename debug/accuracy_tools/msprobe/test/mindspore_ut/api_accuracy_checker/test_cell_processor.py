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
        mock_const.START = "start"
        # 使用 MockCell 代替字符串
        cell = MockCell()
        self.processor.node_hook("prefix", "start")(cell, "input")

        # 验证 cell 的 mindstudio_reserved_name 是否正确设置
        expected_name = "prefix" + Const.SEP + "0"
        self.assertEqual(cell.mindstudio_reserved_name, expected_name)

    @patch('msprobe.core.common.const.Const')
    def test_node_hook_end(self, mock_const):
        mock_const.START = "start"
        cell = MockCell()
        self.processor.node_hook("prefix", "start")(cell, "input")
        self.processor.node_hook("prefix", "stop")(cell, "input", "output")

        # 验证在结束时的操作
        self.assertEqual(len(CellProcessor.cell_stack), 0)
        self.assertIsNone(CellProcessor.api_parent_node)
        self.scope.end_module.assert_called_once_with(cell.mindstudio_reserved_name)


if __name__ == "__main__":
    unittest.main()
