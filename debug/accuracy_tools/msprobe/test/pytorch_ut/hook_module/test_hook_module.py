import unittest
from unittest.mock import MagicMock
import threading

from msprobe.pytorch.hook_module.hook_module import HOOKModule


class TestHOOKModuleInit(unittest.TestCase):

    def setUp(self):
        self.mock_build_hook = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), None))

    def test_thread_handling(self):
        module = HOOKModule(self.mock_build_hook)
        current_thread_id = module.current_thread
        self.assertEqual(current_thread_id, threading.current_thread().ident)

    def test_stop_hook_logic(self):
        module = HOOKModule(self.mock_build_hook)
        self.assertFalse(module.stop_hook)

        HOOKModule.inner_stop_hook[module.current_thread] = True
        module = HOOKModule(self.mock_build_hook)
        self.assertTrue(module.stop_hook)

    def test_prefix_op_name(self):
        HOOKModule.prefix_op_name_ = "test_prefix"
        module = HOOKModule(self.mock_build_hook)
        self.assertEqual(module.prefix, "test_prefix0.")
        self.mock_build_hook.assert_called_once_with(module.prefix)

    def test_hooks_registration(self):
        module = HOOKModule(self.mock_build_hook)
        self.assertTrue(module._forward_pre_hooks)
        self.assertTrue(module._forward_hooks)
        self.assertTrue(module._backward_hooks)
        self.mock_build_hook.assert_called_once_with(module.prefix)







# if __name__ == '__main__':
#     unittest.main()




# class TestHookModule(unittest.TestCase):
    # def test_call_1(self):
    #     def forward_pre_hook():
    #         return "result_input", "result_kwargs"

    #     def forward_hook():
    #         return 2

    #     def backward_hook():
    #         pass

    #     def forward_hook_torch_version_below_2():
    #         pass

    #     def hook(prefix):
    #         return forward_pre_hook, forward_hook, backward_hook, forward_hook_torch_version_below_2
    #     HOOKModule.prefix_op_name_ = "123"
    #     test = HOOKModule(hook)
    #     test._call_func = Mock(return_value=1)
    #     result = test()
    #     self.assertEqual(result, 1)

#     def test_call_2(self):
#         def forward_pre_hook(nope, input, kwargs):
#             return input, kwargs

#         def forward_hook(nope, input, kwargs, result):
#             return input

#         def backward_hook():
#             pass

#         def forward_hook_torch_version_below_2():
#             pass

#         def hook(prefix):
#             return forward_pre_hook, forward_hook, backward_hook, forward_hook_torch_version_below_2
#         HOOKModule.prefix_op_name_ = "123"
#         input = 1
#         test = HOOKModule(hook)

#         def temp_forward(*input, **kwargs):
#             return input

#         test.forward = Mock(return_value=1)
#         result = test(input)
#         self.assertEqual(result, input)
