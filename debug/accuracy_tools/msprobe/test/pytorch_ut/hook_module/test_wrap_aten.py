# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
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

import torch
from msprobe.core.hook_manager import HookSet
from msprobe.pytorch.function_factory import npu_custom_grad_functions
from msprobe.pytorch.hook_module.wrap_aten import (
    AtenOPTemplate,
    white_aten_ops,
    AtenOPPacketTemplate
)


def mock_build_hook(prefix):
    return HookSet(MagicMock(), MagicMock(), MagicMock())


class TestAtenOPTemplate(unittest.TestCase):

    def test_init_with_string_op(self):
        hook = mock_build_hook
        op = 'add'
        template = AtenOPTemplate(op, hook)

        self.assertEqual(template.op, op)
        self.assertTrue(template.prefix_op_name_.startswith('Aten.add'))
        self.assertTrue(template.need_hook)

    def test_init_with_op_overload_packet(self):
        hook = mock_build_hook
        op = MagicMock()
        op._qualified_op_name = 'aten::add'
        op.name.return_value = 'aten::add'
        op._overloadname = 'default'
        template = AtenOPTemplate(op, hook)

        self.assertEqual(template.op, op)
        self.assertTrue(template.prefix_op_name_.startswith('Aten.add'))
        self.assertTrue(template.need_hook)

    def test_forward_with_white_aten_ops(self):
        hook = mock_build_hook
        op = 'add'
        white_aten_ops.append(op)
        template = AtenOPTemplate(op, hook)

        with patch('torch.ops.aten.add', return_value=3) as mock_add:
            result = template.forward(1, 2)
            self.assertEqual(result, 3)
            mock_add.assert_called_once_with(1, 2)

    def test_forward_with_custom_grad_function(self):
        hook = mock_build_hook
        op = 'custom_op'
        npu_custom_grad_functions[op] = MagicMock(return_value=5)
        template = AtenOPTemplate(op, hook)

        result = template.forward(2, 3)
        self.assertEqual(result, 5)

    def test_forward_with_missing_op(self):
        hook = mock_build_hook
        op = 'missing_op'
        template = AtenOPTemplate(op, hook)

        with self.assertRaises(Exception) as context:
            template.forward(1, 2)
        self.assertIn("Skip op[missing_op] accuracy check", str(context.exception))


class TestAtenOPPacketTemplate(unittest.TestCase):

    def setUp(self):
        self.mock_op_packet = MagicMock()
        self.mock_hook = MagicMock()
        self.template = AtenOPPacketTemplate(self.mock_op_packet, self.mock_hook)

    def test_getattr_existing_attribute(self):
        self.mock_op_packet.some_attr = 'test_value'
        self.assertEqual(self.template.some_attr, 'test_value')

    def test_getattr_nonexistent_attribute_raises_attribute_error(self):
        del self.mock_op_packet.nonexistent_attr
        with self.assertRaises(AttributeError) as context:
            _ = self.template.nonexistent_attr
        self.assertIn("or OpOverloadPacket does not have attribute 'nonexistent_attr'.",
                      str(context.exception))

    @patch('msprobe.pytorch.hook_module.wrap_aten.AtenOPTemplate', autospec=True)
    def test_getattr_op_overload(self, MockAtenOPTemplate):
        mock_overload = MagicMock(spec=torch._ops.OpOverload)
        mock_overload._overloadname = 'some_overload_name'

        self.mock_op_packet.some_op = mock_overload
        result = self.template.some_op
        MockAtenOPTemplate.assert_called_with(mock_overload, self.mock_hook)
        self.assertIsInstance(result, AtenOPTemplate)

    @patch('msprobe.pytorch.hook_module.wrap_aten.AtenOPTemplate', return_value=MagicMock())
    def test_call(self, mock_AtenOPTemplate):
        args = ('arg1', 'arg2')
        kwargs = {'key': 'value'}
        self.template(*args, **kwargs)
        mock_AtenOPTemplate.assert_called_with(self.mock_op_packet, self.mock_hook)
        mock_AtenOPTemplate.return_value.assert_called_with(*args, **kwargs)

    def test_overloads(self):
        expected_overloads = ['overload1', 'overload2']
        self.mock_op_packet.overloads.return_value = expected_overloads
        self.assertEqual(self.template.overloads(), expected_overloads)

    def test_getattr_non_opoverload_attribute(self):
        self.mock_op_packet.non_op_attr = 'non_op_value'
        self.assertEqual(self.template.non_op_attr, 'non_op_value')
