import unittest
from unittest.mock import patch, Mock

from msprobe.pytorch.hook_module.hook_module import HOOKModule

class TestHookModule(unittest.TestCase):
    def test_call_1(self):
        def forward_pre_hook():
            return "result_input", "result_kwargs"

        def forward_hook():
            return 2

        def backward_hook():
            pass

        def forward_hook_torch_version_below_2():
            pass

        def hook(prefix):
            return forward_pre_hook, forward_hook, backward_hook, forward_hook_torch_version_below_2
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

        def forward_hook_torch_version_below_2():
            pass

        def hook(prefix):
            return forward_pre_hook, forward_hook, backward_hook, forward_hook_torch_version_below_2
        HOOKModule.prefix_op_name_ = "123"
        input = 2
        test = HOOKModule(hook)

        def temp_forward(*input, **kwargs):
            return input

        test.forward = Mock(return_value=1)
        result = test(input)
        self.assertEqual(result, (input, ))