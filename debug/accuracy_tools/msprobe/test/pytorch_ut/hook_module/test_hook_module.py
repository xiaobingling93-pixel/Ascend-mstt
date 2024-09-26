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
