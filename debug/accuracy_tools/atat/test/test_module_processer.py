import unittest
from atat.pytorch.module_processer import ModuleProcesser
from atat.pytorch.common.utils import Const

import torch

class TestModuleProcesser(unittest.TestCase):
    def test_filter_tensor_and_tuple(self):
        def func(nope, x):
            return x * 2
        
        result_1 = ModuleProcesser.filter_tensor_and_tuple(func)(None, torch.tensor([1]))
        self.assertEqual(result_1, torch.tensor([2]))

        result_2 = ModuleProcesser.filter_tensor_and_tuple(func)(None, "test")
        self.assertEqual(result_2, "test")

    def test_clone_return_value_and_test_clone_if_tensor(self):
        def func(x):
            return x
        
        input = torch.tensor([1])
        input_tuple = (torch.tensor([1]), torch.tensor([2]))
        input_list = [torch.tensor([1]), torch.tensor([2])]
        input_dict = {"A":torch.tensor([1]), "B":torch.tensor([2])}

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

        
    def test_node_hook(self):
        empty_list = []
        test = ModuleProcesser(None)
        pre_hook = test.node_hook("test", Const.START)
        self.assertIsNotNone(pre_hook)
        end_hook = test.node_hook("test", "stop")
        self.assertIsNotNone(end_hook)

        class A():
            pass
        pre_hook(A, None, None)
        self.assertIn("test", test.module_count)
        self.assertFalse(test.module_stack==empty_list)

    def test_module_count_func(self):
        test = ModuleProcesser(None)
        self.assertEqual(test.module_count, {})

        module_name = "nope"
        test.module_count_func(module_name)
        self.assertEqual(test.module_count["nope"], 0)