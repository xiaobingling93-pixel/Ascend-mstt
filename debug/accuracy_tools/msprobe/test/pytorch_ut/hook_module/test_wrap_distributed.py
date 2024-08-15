import unittest
import torch.distributed as dist
from msprobe.pytorch.hook_module.wrap_distributed import *

class TestWrapDistributed(unittest.TestCase):
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
    
    def test_get_distributed_ops(self):
        ops = get_distributed_ops()
        self.assertIsInstance(ops, set)

    def test_DistributedOPTemplate(self):
        self.setUp()
        op_name = 'all_reduce'
        if op_name in get_distributed_ops():
            op = DistributedOPTemplate(op_name, self.hook)
            self.assertEqual(op.op_name_, op_name)

    def test_wrap_distributed_op(self):
        op_name = 'all_reduce'
        if op_name in get_distributed_ops():
            wrapped_op = wrap_distributed_op(op_name, self.hook)
            self.assertTrue(callable(wrapped_op))

    def test_wrap_distributed_ops_and_bind(self):
        wrap_distributed_ops_and_bind(self.hook)
        for op_name in get_distributed_ops():
            self.assertTrue(hasattr(HOOKDistributedOP, "wrap_" + str(op_name)))