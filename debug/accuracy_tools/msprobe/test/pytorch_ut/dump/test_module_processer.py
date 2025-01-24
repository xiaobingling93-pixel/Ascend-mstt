import unittest
from unittest.mock import MagicMock

import torch

from msprobe.core.data_dump.scope import ModuleRangeScope
from msprobe.pytorch.common.utils import Const
from msprobe.pytorch.dump.module_dump.module_processer import ModuleProcesser


class TestModuleProcesser(unittest.TestCase):

    def setUp(self):
        self.mock_tensor = MagicMock(spec=torch.Tensor)
        self.mock_scope = MagicMock()
        self.processor = ModuleProcesser(self.mock_scope)

    def test_scope_is_module_range_scope(self):
        scope = ModuleRangeScope([], [])
        processor = ModuleProcesser(scope)
        self.assertEqual(processor.scope, scope)

    def test_scope_is_not_module_range_scope(self):
        scope = "not a ModuleRangeScope"
        processor = ModuleProcesser(scope)
        self.assertIsNone(processor.scope)

    def test_filter_tensor_and_tuple(self):
        def func(nope, x):
            return x * 2

        result_1 = ModuleProcesser.filter_tensor_and_tuple(func)(None, torch.tensor([1]))
        self.assertEqual(result_1, torch.tensor([2]))
        result_2 = ModuleProcesser.filter_tensor_and_tuple(func)(None, "test")
        self.assertEqual(result_2, "test")

    def test_filter_tensor_and_tuple_with_tensor(self):
        class MockBackwardHook:
            @staticmethod
            def setup_output_hook(*args, **kwargs):
                return args[1]

        mock_hook = MockBackwardHook.setup_output_hook
        wrapped_hook = ModuleProcesser.filter_tensor_and_tuple(mock_hook)

        tensor = torch.tensor([1, 2, 3])
        mock_obj = type('MockObj', (object,), {'tensor_attr': tensor})()
        wrapped_hook(None, mock_obj)
        self.assertIs(mock_obj.tensor_attr, tensor)
        non_tensor_obj = type('MockObj', (object,), {'non_tensor_attr': 'non_tensor_value'})()
        wrapped_hook(None, non_tensor_obj)
        self.assertEqual(non_tensor_obj.non_tensor_attr, 'non_tensor_value')

    def test_clone_return_value_and_test_clone_if_tensor(self):
        def func(x):
            return x

        input = torch.tensor([1])
        input_tuple = (torch.tensor([1]), torch.tensor([2]))
        input_list = [torch.tensor([1]), torch.tensor([2])]
        input_dict = {"A": torch.tensor([1]), "B": torch.tensor([2])}

        result = ModuleProcesser.clone_return_value(func)(input)
        result[0] = 2
        self.assertNotEqual(result, input)
        result_tuple = ModuleProcesser.clone_return_value(func)(input_tuple)
        result_tuple[0][0] = 2
        self.assertNotEqual(result_tuple, input_tuple)
        result_list = ModuleProcesser.clone_return_value(func)(input_list)
        result_list[0][0] = 2
        self.assertNotEqual(result_list, input_list)
        result_dict = ModuleProcesser.clone_return_value(func)(input_dict)
        result_dict["A"][0] = 2
        self.assertNotEqual(result_dict, input_dict)

    def test_module_count_func(self):
        test = ModuleProcesser(None)
        self.assertEqual(test.module_count, {})
        module_name = "nope"
        test.module_count_func(module_name)
        self.assertEqual(test.module_count["nope"], 0)

    def test_node_hook_forward_start(self):
        name_prefix = "forward_layer"
        hook = self.processor.node_hook(name_prefix, start_or_stop=Const.START)
        module = MagicMock()
        input = (self.mock_tensor,)
        module.mindstudio_reserved_name = None
        hook(module, input)
        expected_name = f"forward_layer{Const.SEP}0"
        self.assertEqual(module.mindstudio_reserved_name, [expected_name])
        self.assertIn(expected_name, ModuleProcesser.module_stack)
        self.assertEqual(ModuleProcesser.api_parent_node, expected_name)

    def test_node_hook_forward_stop(self):
        name_prefix = "forward_layer"
        hook = self.processor.node_hook(name_prefix, start_or_stop=Const.STOP)
        ModuleProcesser.module_stack.append(f"forward_layer{Const.SEP}0")

        module = MagicMock()
        input = (self.mock_tensor,)
        reserved_name = f"forward_layer{Const.SEP}0"
        module.mindstudio_reserved_name = [reserved_name]
        hook(module, input)
        self.assertNotIn([f"forward_layer{Const.SEP}0"], ModuleProcesser.module_stack)
        self.assertEqual(ModuleProcesser.api_parent_node, reserved_name)

    def test_node_hook_backward(self):
        name_prefix = "backward_layer"
        hook = self.processor.node_hook(name_prefix, start_or_stop=Const.START)

        module = MagicMock()
        input = (self.mock_tensor,)
        module.mindstudio_reserved_name = None
        ModuleProcesser.module_node[f"forward_layer{Const.SEP}0"] = None
        hook(module, input)
        expected_name = f"backward_layer{Const.SEP}0"
        self.assertEqual(module.mindstudio_reserved_name, [expected_name])
        self.assertIn(expected_name, ModuleProcesser.module_node)

    def test_has_register_backward_hook(self):
        module = MagicMock()
        module._backward_hooks = {0: lambda: None}
        module._is_full_backward_hook = False
        result = self.processor.has_register_backward_hook(module)
        self.assertTrue(result)

        module._is_full_backward_hook = True
        result = self.processor.has_register_backward_hook(module)
        self.assertFalse(result)
