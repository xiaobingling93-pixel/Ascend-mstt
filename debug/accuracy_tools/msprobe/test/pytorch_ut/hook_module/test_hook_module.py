import unittest
from unittest.mock import MagicMock, patch
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


class TestHOOKModuleCall(unittest.TestCase):
    def setUp(self):
        self.mock_build_hook = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), None))
        self.module = HOOKModule(self.mock_build_hook)

    @patch.object(HOOKModule, '_call_func')
    def test_call_function(self, mock_call_func):
        mock_call_func.return_value = "test_result"
        result = self.module("input_data")
        mock_call_func.assert_called_once_with("input_data", **{})
        self.assertEqual(result, "test_result")

    @patch.object(HOOKModule, '_call_func')
    def test_call_func_with_hooks(self, mock_call_func):
        mock_call_func.return_value = "test_result_with_hooks"
        result = self.module("input_data")
        self.assertEqual(result, "test_result_with_hooks")
        HOOKModule.inner_stop_hook[self.module.current_thread] = False
