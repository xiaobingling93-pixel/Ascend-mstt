import unittest
from unittest.mock import patch, MagicMock, call
import multiprocessing
from multiprocessing import Process, Manager, Queue
from msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker import (
    MultiApiAccuracyChecker,
    handle_child_signal,
    handle_main_signal
)
from msprobe.mindspore.api_accuracy_checker.multi_data_manager import MultiDataManager
from msprobe.mindspore.common.log import logger
from mindspore import context
import os
import sys
import signal  # 添加这一行，导入 signal 模块


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

        self.assertIsNone(result)
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

        self.assertIsNone(result)
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

        self.assertIsNone(result)
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

        self.assertIsNone(result)
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

        self.assertIsNone(result)
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

        self.assertIsNone(result)
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

    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.tqdm')
    @patch('multiprocessing.Process')
    @patch('multiprocessing.Queue')
    @patch('msprobe.mindspore.api_accuracy_checker.multi_api_accuracy_checker.logger')
    def test_run_and_compare(self, mock_logger, mock_queue_class, mock_process_class, mock_tqdm):
        # 模拟进程和队列
        # 创建一个假的进度队列
        mock_queue = MagicMock()
        # 设置进度队列的 get 方法，每次返回 1，总共返回 len(self.checker.api_infos) 次
        mock_queue.get.side_effect = [1] * len(self.checker.api_infos)
        mock_queue_class.return_value = mock_queue

        # 创建模拟的进程列表
        mock_processes = []
        for _ in self.args.device_id:
            mock_process = MagicMock()
            mock_process.is_alive.return_value = False  # 模拟进程已完成
            mock_process.exitcode = 0  # 模拟进程正常退出
            mock_process.pid = 12345  # 模拟进程ID
            mock_processes.append(mock_process)

        # 设置 Process 的 side_effect，每次调用返回不同的进程对象
        mock_process_class.side_effect = mock_processes

        # 模拟 tqdm
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar

        # 运行方法
        self.checker.run_and_compare()

        # 验证进程被正确创建
        self.assertEqual(mock_process_class.call_count, len(self.args.device_id))

        # 验证进度条被正确初始化
        mock_tqdm.assert_called_once_with(total=len(self.checker.api_infos), desc="Total Progress", ncols=100)

        # 验证进度队列的 get 方法被正确调用
        self.assertEqual(mock_queue.get.call_count, len(self.checker.api_infos))

        # 验证进度条的 update 方法被正确调用
        self.assertEqual(mock_pbar.update.call_count, len(self.checker.api_infos))

    def test_handle_child_signal(self):
        # 测试 handle_child_signal 函数是否能够正常执行
        signum = signal.SIGINT
        frame = MagicMock()
        try:
            handle_child_signal(signum, frame)
        except Exception as e:
            self.fail(f"handle_child_signal raised an exception {e}")


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
        self.assertIsInstance(self.checker.is_first_write, multiprocessing.managers.ValueProxy)
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
