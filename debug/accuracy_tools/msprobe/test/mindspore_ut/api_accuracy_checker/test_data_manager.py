import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import DataManager
from msprobe.core.common.const import MsCompareConst, CompareConst

class TestDataManager(unittest.TestCase):

    def setUp(self):
        # 设置测试的CSV目录和结果路径
        self.csv_dir = "/tmp/test_csv_dir"
        self.result_csv_path = os.path.join(self.csv_dir, "result.csv")
        self.data_manager = DataManager(self.csv_dir, self.result_csv_path)

    @patch('msprobe.core.common.file_utils.FileOpen', mock_open(read_data="api_name,bench_dtype,shape"))
    def test_initialize_api_names_set(self):
        # 测试初始化 API 名称集合
        self.data_manager.initialize_api_names_set(self.result_csv_path)
        self.assertIn("api_name", self.data_manager.api_names_set)

    def test_is_unique_api(self):
        # 测试唯一 API 名称检测
        self.assertTrue(self.data_manager.is_unique_api("API1"))
        self.assertFalse(self.data_manager.is_unique_api("API1"))

    @patch('os.path.exists', return_value=True)
    @patch('msprobe.core.common.file_utils.check_file_or_directory_path')
    def test_resume_from_last_csv(self, mock_check_file, mock_exists):
        # 测试恢复断点功能
        self.data_manager.resume_from_last_csv(self.result_csv_path)
        self.assertEqual(self.data_manager.csv_dir, os.path.dirname(self.result_csv_path))
        self.assertIsNotNone(self.data_manager.detail_out_path)
        self.assertIsNotNone(self.data_manager.result_out_path)

    @patch('msprobe.core.common.file_utils.write_csv')
    def test_save_results_first_time_write(self, mock_write_csv):
        # 测试首次写入 CSV 表头
        self.data_manager.is_first_write = True
        self.data_manager.save_results("API1")
        mock_write_csv.assert_called()

    @patch('msprobe.core.common.file_utils.FileOpen', mock_open())
    def test_clear_results(self):
        # 测试清除结果
        self.data_manager.results[("API1", "FORWARD")] = ["test_data"]
        self.data_manager.clear_results()
        self.assertEqual(len(self.data_manager.results), 0)

    @patch('msprobe.core.common.file_utils.write_csv')
    def test_record(self):
        # 测试记录数据
        output_list = [("API1", "FORWARD", MagicMock(api_name="API1"), MagicMock())]
        self.data_manager.record(output_list)
        self.assertIn(("API1", "FORWARD"), self.data_manager.results)
        self.assertEqual(len(self.data_manager.results[("API1", "FORWARD")]), 1)

    @patch('msprobe.core.common.file_utils.write_csv')
    def test_to_detail_csv(self, mock_write_csv):
        # 测试生成详细 CSV
        self.data_manager.results = {
            ("API1", "FORWARD"): [(MagicMock(api_name="API1"), {algorithm: MagicMock(compare_value="PASS") for algorithm in ["Algorithm1", "Algorithm2"]})]
        }
        self.data_manager.to_detail_csv(self.result_csv_path)
        mock_write_csv.assert_called()

    @patch('msprobe.core.common.file_utils.write_csv')
    def test_to_result_csv(self, mock_write_csv):
        # 测试生成结果 CSV
        self.data_manager.results = {
            ("API1", "FORWARD"): [(MagicMock(api_name="API1", status=CompareConst.PASS, err_msg=""), {})]
        }
        self.data_manager.to_result_csv(self.result_csv_path)
        mock_write_csv.assert_called()

if __name__ == "__main__":
    unittest.main()
