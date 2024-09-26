import unittest
from unittest.mock import MagicMock, patch

from msprobe.pytorch.hook_module.hook_module import HOOKModule

class TestHOOKModule(unittest.TestCase):
    def setUp(self):
        self.build_hook = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), None))
        self.module = HOOKModule(self.build_hook)
        # HOOKModule.inner_stop_hook = {self.module.current_thread: False}
        # self.module.stop_hook = False  # 确保每个测试的初始状态


    @patch.object(HOOKModule, '_call_func')
    def test_call_function(self, mock_call_func):
        # 模拟 _call_func 返回值
        mock_call_func.return_value = "test_result"
        result = self.module("input_data")

        # 断言 _call_func 被调用
        mock_call_func.assert_called_once_with("input_data", **{})
        # 断言返回值正确
        self.assertEqual(result, "test_result")

    @patch.object(HOOKModule, '_call_func')
    def test_call_func_with_hooks(self, mock_call_func):
        mock_call_func.return_value = "test_result_with_hooks"
        result = self.module("input_data")
        self.assertEqual(result, "test_result_with_hooks")
        HOOKModule.inner_stop_hook[self.module.current_thread] = False


if __name__ == '__main__':
    unittest.main()
