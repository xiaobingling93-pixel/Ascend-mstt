import unittest
from unittest.mock import patch, Mock

from atat.pytorch.hook_module.hook_module import HOOKModule

class TestHookModule(unittest.TestCase):
    def test_call_1(self):
        def forward_pre_hook():
            return "result_input", "result_kwargs"
        def forward_hook():
            return 2
        def backward_hook():
            pass

        def hook(prefix):
            return forward_pre_hook, forward_hook, backward_hook
        HOOKModule.prefix_op_name_ = "123"
        test = HOOKModule(hook)
        test._call_func = Mock(return_value=1)
        result = test()
        self.assertEqual(result, 1)

    def test_call_2(self):
        def forward_pre_hook(nope, input, kwargs):
            return input, kwargs
        def forward_hook(nope, input, kwargs, result):
            return input
        def backward_hook():
            pass

        def hook(prefix):
            return forward_pre_hook, forward_hook, backward_hook
        HOOKModule.prefix_op_name_ = "123"
        input = 2
        test = HOOKModule(hook, input)
        test.forward = Mock(return_value=1)
        result = test()
        self.assertEqual(result, input)