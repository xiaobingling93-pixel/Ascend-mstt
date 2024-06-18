# coding=utf-8
import unittest
import torch
import yaml
from api_accuracy_checker.hook_module.wrap_torch import *

class TestWrapTorch(unittest.TestCase):

    def setUp(self):
        self.op_name = 'add' 
        self.torch_op = wrap_torch_op(self.op_name, self.hook)

    def hook(self, a, b):
        return

    def test_get_torch_ops(self):
        ops = get_torch_ops()
        self.assertIsInstance(ops, set)
        self.assertIn(self.op_name, ops)

    def test_TorchOPTemplate(self):
        template = TorchOPTemplate(self.op_name, self.hook)
        self.assertEqual(template.op_name_, self.op_name)
        self.assertEqual(template.prefix_op_name_, "Torch*" + str(self.op_name) + "*")

    def test_input_param_need_adapt(self):
        template = TorchOPTemplate(self.op_name, self.hook)
        self.assertFalse(template.input_param_need_adapt())

    def test_forward(self):
        template = TorchOPTemplate(self.op_name, self.hook)
        result = template.forward(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
        torch.testing.assert_allclose(result, torch.tensor([5, 7, 9]))

    def test_wrap_torch_ops_and_bind(self):
        wrap_torch_ops_and_bind(self.hook)
        self.assertTrue(hasattr(HOOKTorchOP, "wrap_" + self.op_name))