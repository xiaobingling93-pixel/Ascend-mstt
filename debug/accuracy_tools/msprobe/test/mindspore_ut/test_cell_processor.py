# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import mindspore as ms
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.data_dump.scope import ModuleRangeScope
from msprobe.mindspore.cell_processor import CellProcessor, get_cell_construct
from msprobe.mindspore.common.log import logger


class TestCellProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CellProcessor.reset_cell_stats()
        cls.scope = MagicMock(spec=ModuleRangeScope)
        cls.processor = CellProcessor(cls.scope)

    @classmethod
    def tearDownClass(cls):
        CellProcessor.reset_cell_stats()

    def test_class_attribute(self):
        self.assertTrue(hasattr(CellProcessor, 'cell_count'))
        self.assertTrue(hasattr(CellProcessor, 'cell_stack'))
        self.assertTrue(hasattr(CellProcessor, 'api_parent_node'))
        self.assertTrue(hasattr(CellProcessor, 'module_node'))
        self.assertTrue(hasattr(CellProcessor, 'cell_bw_hook_kernels'))
        self.assertTrue(hasattr(CellProcessor, 'cell_backward_pre_hook'))
        self.assertTrue(hasattr(CellProcessor, 'cell_backward_hook'))

    def test__init(self):
        self.assertIsInstance(self.processor.scope, ModuleRangeScope)
        processor = CellProcessor(None)
        self.assertIsNone(processor.scope)

    def test_get_cell_construct(self):
        def construct(self, *args, **kwargs):
            return len(args)

        _constrct = get_cell_construct(construct)
        ret = _constrct(self, 'argument')
        self.assertFalse(hasattr(self, 'msprobe_input_kwargs'))
        self.assertEqual(ret, 1)

        setattr(self, 'msprobe_hook', True)
        _constrct = get_cell_construct(construct)
        ret = _constrct(self, 'argument')
        self.assertEqual(self.msprobe_input_kwargs, {})
        self.assertEqual(ret, 1)

        del self.msprobe_hook
        del self.msprobe_input_kwargs

    def test_set_and_get_calls_number(self):
        CellProcessor.cell_count = {}
        count = self.processor.set_and_get_calls_number("cell")
        self.assertEqual(count, 0)
        self.assertEqual(CellProcessor.cell_count["cell"], 0)

        count = self.processor.set_and_get_calls_number("cell")
        self.assertEqual(count, 1)
        self.assertEqual(CellProcessor.cell_count["cell"], 1)

        CellProcessor.cell_count = {}

    def test_reset_cell_stats(self):
        CellProcessor.cell_count['cell'] = 0
        CellProcessor.cell_stack.append('cell')
        CellProcessor.api_parent_node = 'cell'
        CellProcessor.module_node['cell'] = 'null'
        CellProcessor.cell_bw_hook_kernels['cell'] = 'bw'
        CellProcessor.cell_backward_pre_hook.append('backward_pre_hook')
        CellProcessor.cell_backward_hook.append('backward_hook')

        CellProcessor.reset_cell_stats()
        self.assertEqual(CellProcessor.cell_count, {})
        self.assertEqual(CellProcessor.cell_stack, [])
        self.assertIsNone(CellProcessor.api_parent_node)
        self.assertEqual(CellProcessor.module_node, {})
        self.assertEqual(CellProcessor.cell_bw_hook_kernels, {})
        self.assertEqual(CellProcessor.cell_backward_pre_hook, [])
        self.assertEqual(CellProcessor.cell_backward_hook, [])

    def test_register_cell_hook(self):
        with self.assertRaises(MsprobeException) as context:
            self.processor.register_cell_hook([], None)
        self.assertEqual(str(context.exception), '[msprobe] 无效参数：The model cannot be None, when level is "L0" or "mix"')

        with patch('msprobe.mindspore.cell_processor.is_mindtorch') as mock_is_mindtorch, \
             patch('msprobe.mindspore.cell_processor.get_cells_and_names') as mock_get_cells_and_names, \
             patch('msprobe.mindspore.cell_processor.CellProcessor.build_cell_hook') as mock_build_cell_hook, \
             patch('msprobe.mindspore.cell_processor.get_cell_construct') as mock_get_cell_construct, \
             patch.object(logger, 'info') as mock_logger_info:
            mock_cell = MagicMock()
            mock_sub_cell = MagicMock()
            mock_get_cells_and_names.return_value = {'-1': [('cell', mock_cell), ('sub_cell', mock_sub_cell)]}
            mock_build_cell_hook.return_value = ('forward_pre_hook', 'forward_hook')
            mock_get_cell_construct.return_value = '_construct'

            mock_is_mindtorch.return_value = False
            setattr(MagicMock, 'construct', 'construct')
            self.processor.register_cell_hook(mock_cell, None)
            self.assertTrue(mock_sub_cell.__class__.msprobe_construct)
            mock_get_cell_construct.assert_called_with('construct')
            self.assertEqual(mock_sub_cell.__class__.construct, '_construct')
            self.assertTrue(mock_sub_cell.msprobe_hook)
            mock_build_cell_hook.assert_called_with('Cell.sub_cell.MagicMock.', None)
            mock_cell.assert_not_called()
            mock_sub_cell.register_forward_pre_hook.assert_called_with('forward_pre_hook')
            mock_sub_cell.register_forward_hook.assert_called_with('forward_hook')
            mock_logger_info.assert_called_with('The cell hook function is successfully mounted to the model.')

            del MagicMock.construct
            del mock_sub_cell.__class__.construct
            del mock_sub_cell.__class__.msprobe_construct

            mock_get_cell_construct.reset_mock()
            mock_another_sub_cell = MagicMock()
            setattr(mock_another_sub_cell.__class__, 'msprobe_construct', True)
            mock_get_cells_and_names.return_value = {'-1': [('cell', mock_cell),
                                                            ('another_sub_cell', mock_another_sub_cell)]}
            self.processor.register_cell_hook(mock_cell, None)
            mock_get_cell_construct.assert_not_called()
            mock_another_sub_cell.register_forward_pre_hook.assert_called_with('forward_pre_hook')
            mock_another_sub_cell.register_forward_hook.assert_called_with('forward_hook')

            del mock_another_sub_cell.__class__.msprobe_construct

            mock_build_cell_hook.reset_mock()
            mock_get_cell_construct.reset_mock()
            mock_another_sub_cell.reset_mock()
            setattr(MagicMock, 'forward', 'forward')
            mock_is_mindtorch.return_value = True
            self.processor.register_cell_hook(mock_cell, None)
            self.assertTrue(mock_another_sub_cell.__class__.msprobe_construct)
            mock_get_cell_construct.assert_called_with('forward')
            mock_build_cell_hook.assert_called_with('Module.another_sub_cell.MagicMock.', None)
            mock_cell.assert_not_called()
            mock_another_sub_cell.register_forward_pre_hook.assert_called_with('forward_pre_hook')
            mock_another_sub_cell.register_forward_hook.assert_called_with('forward_hook')

            del MagicMock.forward
            del mock_another_sub_cell.__class__.forward
            del mock_another_sub_cell.__class__.msprobe_construct

    def test_build_cell_hook(self):
        CellProcessor.reset_cell_stats()

        cell_name = 'Cell.cell.Cell.'
        mock_build_data_hook = MagicMock()
        mock_backward_data_hook = MagicMock()
        target_grad_output = (Tensor([0.5]),)
        mock_backward_data_hook.return_value = target_grad_output
        mock_build_data_hook.return_value = (None, None, mock_backward_data_hook, None)
        mock_cell = MagicMock()

        with patch.object(_inner_ops, 'CellBackwardHook') as mock_CellBackwardHook:
            forward_pre_hook, forward_hook = self.processor.build_cell_hook(cell_name, mock_build_data_hook)

            mock_bw = mock_CellBackwardHook.return_value
            mock_bw.return_value = (Tensor([0.0]),)
            args = (Tensor([1.0]),)
            target_args = (Tensor([0.0]),)
            full_forward_name = f'{cell_name}{Const.FORWARD}.0'
            full_backward_name = f'{cell_name}{Const.BACKWARD}.0'

            ret = forward_pre_hook(mock_cell, args)
            self.assertIsNone(CellProcessor.module_node[full_forward_name])
            self.assertEqual(CellProcessor.cell_stack, [full_forward_name])
            self.assertEqual(CellProcessor.api_parent_node, full_forward_name)
            self.scope.begin_module.assert_called_with(full_forward_name)
            mock_build_data_hook.assert_called_with('Module', full_forward_name)
            self.assertEqual(len(CellProcessor.cell_backward_hook), 1)
            mock_CellBackwardHook.assert_called_with(full_backward_name, mock_cell, CellProcessor.cell_backward_hook[-1])
            mock_bw.register_backward_hook.assert_called_once()
            self.assertTrue((ret[0] == target_args[0]).all())

            backward_hook = CellProcessor.cell_backward_hook[-1][full_backward_name]
            grad_input = (Tensor([1.0]),)
            grad_output = (Tensor([2.0]),)
            ret = backward_hook(mock_cell, grad_input, grad_output)
            mock_backward_data_hook.assert_called_with(mock_cell, grad_input, grad_output)
            self.assertFalse(mock_cell.has_pre_hook_called)
            self.assertEqual(CellProcessor.cell_stack, [])
            self.assertIsNone(CellProcessor.api_parent_node)
            self.scope.end_module.assert_called_with(full_backward_name)
            self.assertTrue((ret[0] == target_grad_output[0]).all())

            mock_build_data_hook.reset_mock()
            args = (Tensor([1], dtype=ms.int32),)
            full_forward_name = f'{cell_name}{Const.FORWARD}.1'
            ret = forward_pre_hook(mock_cell, args)
            self.assertIsNone(CellProcessor.module_node[full_forward_name])
            self.assertEqual(CellProcessor.cell_stack, [full_forward_name])
            self.assertEqual(CellProcessor.api_parent_node, full_forward_name)
            self.scope.begin_module.assert_called_with(full_forward_name)
            self.assertEqual(len(CellProcessor.cell_backward_hook), 1)
            mock_build_data_hook.assert_not_called()

            full_forward_name = f'{cell_name}{Const.FORWARD}.0'
            CellProcessor.cell_count = {cell_name: 0}
            CellProcessor.cell_stack = [full_forward_name]
            CellProcessor.api_parent_node = full_forward_name
            CellProcessor.module_node = {full_forward_name: None}
            mock_CellBackwardHook.reset_mock()
            mock_bw.reset_mock()
            mock_backward_data_hook.reset_mock()
            mock_forward_data_hook_hook = MagicMock()
            target_output = Tensor([0.5])
            mock_forward_data_hook_hook.return_value = target_output
            mock_build_data_hook.return_value = (None, mock_forward_data_hook_hook, mock_backward_data_hook, None)
            args = (Tensor([1.0]),)
            output = Tensor([2.0])
            ret = forward_hook(mock_cell, args, output)
            self.assertTrue((ret == target_output).all())

    def test_set_construct_info_in_pre_hook(self):
        CellProcessor.reset_cell_stats()
        self.processor.set_construct_info_in_pre_hook('full_name')
        self.assertEqual(CellProcessor.module_node['full_name'], None)
        self.assertEqual(CellProcessor.cell_stack, ['full_name'])
        self.assertEqual(CellProcessor.api_parent_node, 'full_name')
        self.scope.begin_module.assert_called_with('full_name')

        self.scope.begin_module.reset_mock()
        self.processor.set_construct_info_in_pre_hook('sub_cell_name')
        self.assertEqual(CellProcessor.module_node, {'full_name': None, 'sub_cell_name': 'full_name'})
        self.assertEqual(CellProcessor.cell_stack, ['full_name', 'sub_cell_name'])
        self.assertEqual(CellProcessor.api_parent_node, 'sub_cell_name')
        self.scope.begin_module.assert_called_with('sub_cell_name')

        CellProcessor.reset_cell_stats()

    def test_set_construct_info_in_hook(self):
        CellProcessor.reset_cell_stats()
        self.processor.set_construct_info_in_hook('full_name')
        self.assertIsNone(CellProcessor.api_parent_node)
        self.scope.end_module.assert_called_with('full_name')

        self.scope.end_module.reset_mock()
        CellProcessor.cell_stack = ['full_name']
        self.processor.set_construct_info_in_hook('full_name')
        self.assertEqual(CellProcessor.cell_stack, [])
        self.assertIsNone(CellProcessor.api_parent_node)
        self.scope.end_module.assert_called_with('full_name')

        self.scope.end_module.reset_mock()
        CellProcessor.cell_stack = ['Cell.0', 'Cell.1']
        self.processor.set_construct_info_in_hook('full_name')
        self.assertEqual(CellProcessor.cell_stack, ['Cell.0'])
        self.assertEqual(CellProcessor.api_parent_node, 'Cell.0')
        self.scope.end_module.assert_called_with('full_name')

        CellProcessor.reset_cell_stats()
