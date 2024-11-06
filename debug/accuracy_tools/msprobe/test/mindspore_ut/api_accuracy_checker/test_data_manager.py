import unittest
import os
import csv
from unittest.mock import patch, mock_open, MagicMock
from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import DataManager
from msprobe.core.common.const import MsCompareConst, CompareConst


class TestDataManager(unittest.TestCase):

    def setUp(self):
        # 设置测试数据的路径和参数
        self.csv_dir = "/tmp/test_csv_dir"
        self.result_csv_path = os.path.join(self.csv_dir, "result.csv")
        self.data_manager = DataManager(self.csv_dir, self.result_csv_path)

    @patch("os.path.exists", return_value=True)
    @patch("msprobe.core.common.file_utils.FileOpen")
    def test_initialize_api_names_set(self, mock_file_open, mock_exists):
        # Mock CSV内容
        mock_file_open.return_value.__enter__.return_value = iter([
            [MsCompareConst.DETAIL_CSV_API_NAME, "TestColumn"],
            ["API1", ""],
            ["API2", ""]
        ])

        self.data_manager.initialize_api_names_set(self.result_csv_path)

        # 检查是否成功读取 API 名称集合
        self.assertIn("API1", self.data_manager.api_names_set)
        self.assertIn("API2", self.data_manager.api_names_set)
        self.assertEqual(len(self.data_manager.api_names_set), 2)

    def test_is_unique_api(self):
        # 检查是否返回唯一的 API 名称
        self.assertTrue(self.data_manager.is_unique_api("API1"))
        self.assertFalse(self.data_manager.is_unique_api("API1"))  # 再次添加相同名称，应返回 False

    @patch("os.path.exists", return_value=True)
    @patch("msprobe.core.common.file_utils.check_file_or_directory_path")
    def test_resume_from_last_csv(self, mock_check_file, mock_exists):
        self.data_manager.resume_from_last_csv(self.result_csv_path)

        # 检查是否正确初始化 CSV 路径
        self.assertEqual(self.data_manager.csv_dir, os.path.dirname(self.result_csv_path))
        self.assertIsNotNone(self.data_manager.detail_out_path)
        self.assertIsNotNone(self.data_manager.result_out_path)

    @patch("msprobe.core.common.file_utils.write_csv")
    def test_save_results_first_time_write(self, mock_write_csv):
        # 初次写入，应设置表头
        self.data_manager.is_first_write = True
        self.data_manager.save_results("API1")

        # 检查是否调用了写入表头函数
        mock_write_csv.assert_called()

    def test_record(self):
        # 模拟数据输入
        output_list = [("API1", "FORWARD", MagicMock(api_name="API1"), MagicMock())]
        self.data_manager.record(output_list)

        # 检查是否成功记录
        self.assertIn(("API1", "FORWARD"), self.data_manager.results)
        self.assertEqual(len(self.data_manager.results[("API1", "FORWARD")]), 1)

    def test_clear_results(self):
        # 添加测试数据到 results 中
        self.data_manager.results[("API1", "FORWARD")] = ["test_data"]
        self.data_manager.clear_results()

        # 检查是否清除结果
        self.assertEqual(len(self.data_manager.results), 0)

    @patch("msprobe.core.common.file_utils.write_csv")
    def test_to_detail_csv(self, mock_write_csv):
        self.data_manager.results = {
            ("API1", "FORWARD"): [(MagicMock(api_name="API1"),
                                   {algorithm: MagicMock(compare_value="PASS") for algorithm in
                                    ["Algorithm1", "Algorithm2"]})]
        }
        self.data_manager.to_detail_csv(self.result_csv_path)
        mock_write_csv.assert_called()

    @patch("msprobe.core.common.file_utils.write_csv")
    def test_to_result_csv(self, mock_write_csv):
        self.data_manager.results = {
            ("API1", "FORWARD"): [(MagicMock(api_name="API1", status=CompareConst.PASS, err_msg=""), {})]
        }
        self.data_manager.to_result_csv(self.result_csv_path)
        mock_write_csv.assert_called()


if __name__ == "__main__":
    unittest.main()
