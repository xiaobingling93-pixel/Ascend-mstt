import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import csv
from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import DataManager
from msprobe.core.common.const import MsCompareConst, CompareConst


class TestDataManager(unittest.TestCase):

    @patch('msprobe.core.common.file_utils.FileOpen', mock_open(read_data="api_name,bench_dtype,shape"))
    def test_initialize_api_names_set(self):
        # 测试是否成功初始化 API 名称集合
        data_manager = DataManager(csv_dir="fake_dir", result_csv_path="fake_path")
        data_manager.initialize_api_names_set("fake_path")

        # 检查是否正确存储 API 名称
        self.assertIn("api_name", data_manager.api_names_set)

    def test_is_unique_api(self):
        # 测试是否能正确识别唯一的 API 名称
        data_manager = DataManager(csv_dir="fake_dir", result_csv_path="fake_path")
        self.assertTrue(data_manager.is_unique_api("API1"))
        self.assertFalse(data_manager.is_unique_api("API1"))  # 重复添加，应返回 False

    @patch('os.path.exists', return_value=True)
    @patch('msprobe.core.common.file_utils.check_file_or_directory_path')
    def test_resume_from_last_csv(self, mock_check_file, mock_exists):
        data_manager = DataManager(csv_dir="fake_dir", result_csv_path="fake_path")
        data_manager.resume_from_last_csv("fake_path")

        # 检查是否正确设置路径
        self.assertEqual(data_manager.csv_dir, os.path.dirname("fake_path"))
        self.assertIsNotNone(data_manager.detail_out_path)
        self.assertIsNotNone(data_manager.result_out_path)

    @patch('msprobe.core.common.file_utils.write_csv')
    def test_save_results_first_time_write(self, mock_write_csv):
        # 模拟首次写入 CSV 表头
        data_manager = DataManager(csv_dir="fake_dir", result_csv_path="fake_path")
        data_manager.is_first_write = True
        data_manager.save_results("API1")

        # 检查是否成功调用写入 CSV 表头
        mock_write_csv.assert_called()

    @patch('msprobe.core.common.file_utils.FileOpen', mock_open())
    def test_clear_results(self):
        # 测试清除结果功能
        data_manager = DataManager(csv_dir="fake_dir", result_csv_path="fake_path")
        data_manager.results[("API1", "FORWARD")] = ["test_data"]
        data_manager.clear_results()

        # 检查结果是否被清空
        self.assertEqual(len(data_manager.results), 0)

    @patch('msprobe.core.common.file_utils.write_csv')
    def test_record(self):
        # 模拟记录数据
        data_manager = DataManager(csv_dir="fake_dir", result_csv_path="fake_path")
        output_list = [("API1", "FORWARD", MagicMock(api_name="API1"), MagicMock())]
        data_manager.record(output_list)

        # 检查记录是否成功添加
        self.assertIn(("API1", "FORWARD"), data_manager.results)
        self.assertEqual(len(data_manager.results[("API1", "FORWARD")]), 1)

    @patch('msprobe.core.common.file_utils.write_csv')
    def test_to_detail_csv(self, mock_write_csv):
        # 测试生成详细 CSV
        data_manager = DataManager(csv_dir="fake_dir", result_csv_path="fake_path")
        data_manager.results = {
            ("API1", "FORWARD"): [(MagicMock(api_name="API1"),
                                   {algorithm: MagicMock(compare_value="PASS") for algorithm in
                                    ["Algorithm1", "Algorithm2"]})]
        }
        data_manager.to_detail_csv("fake_path")

        # 检查写入是否成功
        mock_write_csv.assert_called()

    @patch('msprobe.core.common.file_utils.write_csv')
    def test_to_result_csv(self, mock_write_csv):
        # 测试生成结果 CSV
        data_manager = DataManager(csv_dir="fake_dir", result_csv_path="fake_path")
        data_manager.results = {
            ("API1", "FORWARD"): [(MagicMock(api_name="API1", status=CompareConst
