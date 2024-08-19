import unittest
import torch
from msprobe.pytorch.hook_module.wrap_torch import *

class TestWrapTorch(unittest.TestCase):

    def hook(name, prefix):
        def forward_pre_hook(nope, input, kwargs):
            return input, kwargs

        def forward_hook(nope, input, kwargs, result):
            return 2

        def backward_hook():
            pass

        def forward_hook_torch_version_below_2():
            pass

        return forward_pre_hook, forward_hook, backward_hook, forward_hook_torch_version_below_2
    
    def setUp(self):

        self.op_name = 'add'
        self.torch_op = wrap_torch_op(self.op_name, self.hook)

    def test_get_torch_ops(self):
        self.setUp()
        ops = get_torch_ops()
        self.assertIsInstance(ops, set)
        self.assertIn(self.op_name, ops)

    def test_TorchOPTemplate(self):
        self.setUp()
        template = TorchOPTemplate(self.op_name, self.hook)
        self.assertEqual(template.op_name_, self.op_name)        
        self.assertEqual(template.prefix_op_name_, "Torch." + str(self.op_name) + ".")

    def test_forward(self):
        self.setUp()
        template = TorchOPTemplate(self.op_name, self.hook)
        result = template.forward(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
        torch.testing.assert_close(result, torch.tensor([5, 7, 9]))

    def test_wrap_torch_ops_and_bind(self):
        self.setUp()
        wrap_torch_ops_and_bind(self.hook)
        self.assertTrue(hasattr(HOOKTorchOP, "wrap_" + self.op_name))