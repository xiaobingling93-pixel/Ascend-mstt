import unittest
import torch
import yaml
from atat.pytorch.hook_module.wrap_tensor import get_tensor_ops, HOOKTensor, TensorOPTemplate, wrap_tensor_op, wrap_tensor_ops_and_bind

class TestWrapTensor(unittest.TestCase):

    def hook(name, prefix):
        def forward_pre_hook(nope, input, kwargs):
            return input, kwargs
        def forward_hook(nope, input, kwargs, result):
            return 2
        def backward_hook():
            pass
        return forward_pre_hook, forward_hook, backward_hook
    
    def test_get_tensor_ops(self):
        result = get_tensor_ops()
        self.assertIsInstance(result, set)

    def test_HOOKTensor(self):
        hook_tensor = HOOKTensor()
        self.assertIsInstance(hook_tensor, HOOKTensor)

    def test_TensorOPTemplate(self):
        tensor_op_template = TensorOPTemplate('add', self.hook)
        self.assertTrue(tensor_op_template.op_name_, 'add')

    def test_wrap_tensor_op(self):
        wrapped_op = wrap_tensor_op('add', self.hook)
        self.assertTrue(callable(wrapped_op))

    def test_wrap_tensor_ops_and_bind(self):
        wrap_tensor_ops_and_bind(self.hook)
        self.assertTrue(hasattr(HOOKTensor, 'wrap_add'))