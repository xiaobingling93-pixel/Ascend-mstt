# Python 标准库
import os
import sys
import signal  # 导入 signal 模块
import multiprocessing
from multiprocessing import Process, Manager, Queue

# 第三方库
import unittest
from unittest.mock import patch, MagicMock, call

# 自定义模块
from msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker import (
    MultiApiAccuracyChecker
)
from msprobe.mindspore.api_accuracy_checker.multi_data_manager import MultiDataManager
from msprobe.mindspore.common.log import logger
from mindspore import context
from msprobe.core.common.const import Const


class TestMultiApiAccuracyChecker(unittest.TestCase):
    def setUp(self):
        # 初始化参数
        self.args = MagicMock()
        self.args.out_path = "./test_output"
        self.args.result_csv_path = "./test_output/result.csv"
        self.args.device_id = [0, 1]  # 模拟两个设备ID

        # **创建测试输出目录**
        if not os.path.exists(self.args.out_path):
            os.makedirs(self.args.out_path)

        # **创建空的 result.csv 文件**
        with open(self.args.result_csv_path, 'w') as f:
            f.write("API Name,Forward Test Success,Backward Test Success,Message\n")  # 写入表头


        # 创建 MultiApiAccuracyChecker 实例
        self.checker = MultiApiAccuracyChecker(self.args)

        # 模拟 api_infos 数据
        self.checker.api_infos = {
            'API_1': MagicMock(),
            'API_2': MagicMock(),
            'API_3': MagicMock(),
            'API_4': MagicMock(),
        }


    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.logger')
    def test_process_forward_no_forward_info(self, mock_logger):
        """测试当 check_forward_info 返回 False 时，process_forward 返回 None 并记录调试日志。"""
        # 设置 current_device_id
        self.checker.current_device_id = 0

        api_info = MagicMock()
        api_info.check_forward_info.return_value = False

        result = self.checker.process_forward("API_1", api_info)

        self.assertEqual(result, Const.EXCEPTION_NONE)
        mock_logger.debug.assert_called_with(
            "[Device 0] API: API_1 lacks forward information, skipping forward check."
        )

    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.logger')
    def test_process_forward_prepare_exception(self, mock_logger):
        """测试当 prepare_api_input_aggregation 抛出异常时，process_forward 返回 None 并记录警告日志。"""
        # 设置 current_device_id
        self.checker.current_device_id = 0

        api_info = MagicMock()
        api_info.check_forward_info.return_value = True

        with patch.object(self.checker, 'prepare_api_input_aggregation', side_effect=Exception("Test exception")):
            result = self.checker.process_forward("API_1", api_info)

        self.assertEqual(result, Const.EXCEPTION_NONE)
        mock_logger.warning.assert_called_with(
            "[Device 0] Exception occurred while getting forward API inputs for API_1. Skipping forward check. Detailed exception information: Test exception."
        )

    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.logger')
    def test_process_forward_run_and_compare_exception(self, mock_logger):
        """测试当 run_and_compare_helper 抛出异常时，process_forward 返回 None 并记录警告日志。"""
        # 设置 current_device_id
        self.checker.current_device_id = 0

        api_info = MagicMock()
        api_info.check_forward_info.return_value = True

        forward_inputs_aggregation = MagicMock()

        with patch.object(self.checker, 'prepare_api_input_aggregation', return_value=forward_inputs_aggregation), \
             patch.object(self.checker, 'run_and_compare_helper', side_effect=Exception("Test exception")):
            result = self.checker.process_forward("API_1", api_info)

        self.assertEqual(result, Const.EXCEPTION_NONE)
        mock_logger.warning.assert_called_with(
            "[Device 0] Exception occurred while running and comparing API_1 forward API. Detailed exception information: Test exception."
        )

    def test_process_forward_success(self):
        """测试 process_forward 成功执行时，返回正确的输出列表。"""
        # 设置 current_device_id
        self.checker.current_device_id = 0

        api_info = MagicMock()
        api_info.check_forward_info.return_value = True

        forward_inputs_aggregation = MagicMock()
        forward_output_list = MagicMock()

        with patch.object(self.checker, 'prepare_api_input_aggregation', return_value=forward_inputs_aggregation), \
             patch.object(self.checker, 'run_and_compare_helper', return_value=forward_output_list):
            result = self.checker.process_forward("API_1", api_info)

        self.assertEqual(result, forward_output_list)

    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.logger')
    def test_process_backward_no_backward_info(self, mock_logger):
        """测试当 check_backward_info 返回 False 时，process_backward 返回 None 并记录调试日志。"""
        # 设置 current_device_id
        self.checker.current_device_id = 1

        api_info = MagicMock()
        api_info.check_backward_info.return_value = False

        result = self.checker.process_backward("API_2", api_info)

        self.assertEqual(result, Const.EXCEPTION_NONE)
        mock_logger.debug.assert_called_with(
            "[Device 1] API: API_2 lacks backward information, skipping backward check."
        )

    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.logger')
    def test_process_backward_prepare_exception(self, mock_logger):
        """测试当 prepare_api_input_aggregation 抛出异常时，process_backward 返回 None 并记录警告日志。"""
        # 设置 current_device_id
        self.checker.current_device_id = 1

        api_info = MagicMock()
        api_info.check_backward_info.return_value = True

        with patch.object(self.checker, 'prepare_api_input_aggregation', side_effect=Exception("Test exception")):
            result = self.checker.process_backward("API_2", api_info)

        self.assertEqual(result, Const.EXCEPTION_NONE)
        mock_logger.warning.assert_called_with(
            "[Device 1] Exception occurred while getting backward API inputs for API_2. Skipping backward check. Detailed exception information: Test exception."
        )

    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.logger')
    def test_process_backward_run_and_compare_exception(self, mock_logger):
        """测试当 run_and_compare_helper 抛出异常时，process_backward 返回 None 并记录警告日志。"""
        # 设置 current_device_id
        self.checker.current_device_id = 1

        api_info = MagicMock()
        api_info.check_backward_info.return_value = True

        backward_inputs_aggregation = MagicMock()

        with patch.object(self.checker, 'prepare_api_input_aggregation', return_value=backward_inputs_aggregation), \
             patch.object(self.checker, 'run_and_compare_helper', side_effect=Exception("Test exception")):
            result = self.checker.process_backward("API_2", api_info)

        self.assertEqual(result, Const.EXCEPTION_NONE)
        mock_logger.warning.assert_called_with(
            "[Device 1] Exception occurred while running and comparing API_2 backward API. Detailed exception information: Test exception."
        )

    def test_process_backward_success(self):
        """测试 process_backward 成功执行时，返回正确的输出列表。"""
        # 设置 current_device_id
        self.checker.current_device_id = 1

        api_info = MagicMock()
        api_info.check_backward_info.return_value = True

        backward_inputs_aggregation = MagicMock()
        backward_output_list = MagicMock()

        with patch.object(self.checker, 'prepare_api_input_aggregation', return_value=backward_inputs_aggregation), \
             patch.object(self.checker, 'run_and_compare_helper', return_value=backward_output_list):
            result = self.checker.process_backward("API_2", api_info)

        self.assertEqual(result, backward_output_list)

    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.context')
    def test_process_on_device_api_not_unique(self, mock_context):
        # 测试当 API 不是唯一时的行为
        with patch.object(self.checker.multi_data_manager, 'is_unique_api', return_value=False) as mock_is_unique_api, \
             patch.object(self.checker, 'process_forward') as mock_process_forward, \
             patch.object(self.checker, 'process_backward') as mock_process_backward:

            device_id = 0
            api_infos = [('API_1', MagicMock())]
            progress_queue = Queue()

            self.checker.process_on_device(device_id, api_infos, progress_queue)

            # 验证 process_forward 和 process_backward 未被调用
            mock_process_forward.assert_not_called()
            mock_process_backward.assert_not_called()

    def test_init(self):
        # 测试初始化方法
        self.assertIsInstance(self.checker.manager, Manager().__class__)
        self.assertIsInstance(self.checker.multi_data_manager, MultiDataManager)
        self.assertEqual(self.checker.args, self.args)

    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.context')
    def test_process_on_device_no_output(self, mock_context):
        # 测试当 forward 和 backward 返回 None 时的行为
        with patch.object(self.checker.multi_data_manager, 'is_unique_api', return_value=True), \
             patch.object(self.checker.multi_data_manager, 'record') as mock_record, \
             patch.object(self.checker.multi_data_manager, 'save_results') as mock_save_results, \
             patch.object(self.checker, 'process_forward', return_value=None), \
             patch.object(self.checker, 'process_backward', return_value=None):

            device_id = 0
            api_infos = [('API_1', MagicMock())]
            progress_queue = Queue()

            self.checker.process_on_device(device_id, api_infos, progress_queue)

            # 验证 record 未被调用
            mock_record.assert_not_called()

            # 验证 save_results 被调用
            mock_save_results.assert_called_once_with('API_1')

    def tearDown(self):
        # 清理资源
        if hasattr(self.checker, 'manager'):
            self.checker.manager.shutdown()
        # 清理测试输出目录
        if os.path.exists(self.args.out_path):
            for root, dirs, files in os.walk(self.args.out_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.args.out_path)

if __name__ == '__main__':
    unittest.main()
