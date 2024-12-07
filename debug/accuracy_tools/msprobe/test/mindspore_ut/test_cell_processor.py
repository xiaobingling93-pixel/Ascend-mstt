# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import MagicMock, patch

from msprobe.core.common.const import Const
from msprobe.core.data_dump.scope import ModuleRangeScope
from msprobe.mindspore.cell_processor import CellProcessor


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
        mock_const.SEP = "."  # 确保 SEPARATOR 设置为字符串
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

    def test_set_and_get_reserved_name(self):
        cell = MockCell()
        cell.mindstudio_reserved_name = "mindstudio_reserved_name"
        CellProcessor.reset_cell_stats()

        cell_name = "Cell.net.Net.forward"
        ret = self.processor.set_and_get_reserved_name(cell, cell_name)
        self.assertEqual(ret, cell_name + Const.SEP + "0")
        self.assertEqual(cell.mindstudio_reserved_name, ret)
        self.assertEqual(CellProcessor.cell_count[cell_name], 0)
        self.assertFalse(hasattr(cell, "has_pre_hook_called"))

        cell.has_pre_hook_called = False
        ret = self.processor.set_and_get_reserved_name(cell, cell_name)
        self.assertEqual(ret, cell_name + Const.SEP + "1")
        self.assertEqual(cell.mindstudio_reserved_name, ret)
        self.assertEqual(CellProcessor.cell_count[cell_name], 1)
        self.assertFalse(cell.has_pre_hook_called)

        cell.has_pre_hook_called = True
        cell.mindstudio_reserved_name = "mindstudio_reserved_name"
        CellProcessor.reset_cell_stats()
        ret = self.processor.set_and_get_reserved_name(cell, cell_name)
        self.assertEqual(ret, "mindstudio_reserved_name")
        self.assertEqual(cell.mindstudio_reserved_name, ret)
        self.assertEqual(CellProcessor.cell_count, {})
        self.assertFalse(cell.has_pre_hook_called)

        ret = self.processor.set_and_get_reserved_name(cell, cell_name, is_called_by_pre_hook=True)
        self.assertEqual(ret, cell_name + Const.SEP + "0")
        self.assertEqual(cell.mindstudio_reserved_name, ret)
        self.assertEqual(CellProcessor.cell_count[cell_name], 0)
        self.assertTrue(cell.has_pre_hook_called)
        CellProcessor.reset_cell_stats()
