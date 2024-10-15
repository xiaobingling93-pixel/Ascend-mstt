import unittest
from unittest.mock import patch
import os
import shutil
import numpy as np

from msprobe.pytorch.parse_tool.lib.compare import Compare


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.compare = Compare()
        self.my_dump_path = './my_path'
        self.golden_path = './golden_path'
        self.result_dir = './result_dir'
        self.csv_path = './my.csv'
        self.msaccucmp_path = './msaccucmp.py'
        self.npy_file1 = './file1.npy'
        self.npy_file2 = './file2.npy'
        self.var1 = np.array([1, 2, 3, 4, 5])
        self.var2 = np.array([2, 3, 4, 5, 6])
        np.save(self.npy_file1, self.var1)
        np.save(self.npy_file2, self.var2)

    def tearDown(self):
        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir)
        if os.path.exists(self.npy_file1):
            os.remove(self.npy_file1)
        if os.path.exists(self.npy_file2):
            os.remove(self.npy_file2)
        if os.path.exists(self.npy_file1 + '.txt'):
            os.remove(self.npy_file1 + '.txt')
        if os.path.exists(self.npy_file2 + '.txt'):
            os.remove(self.npy_file2 + '.txt')
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    @patch('msprobe.pytorch.parse_tool.lib.compare.Compare.compare_vector')
    def test_npu_vs_npu_compare(self, mock_comare_vector):
        self.compare.npu_vs_npu_compare(self.my_dump_path, self.golden_path, self.result_dir, self.msaccucmp_path)

        mock_comare_vector.assert_called_once()

    def test_compare_vector(self):
        with patch('msprobe.pytorch.parse_tool.lib.compare.Util.check_path_valid', return_value=True), \
            patch('msprobe.pytorch.parse_tool.lib.compare.Util.check_msaccucmp', return_value=None), \
            patch('msprobe.pytorch.parse_tool.lib.compare.Util.execute_command', return_value=0):
            res = self.compare.compare_vector(self.my_dump_path, self.golden_path, self.result_dir, self.msaccucmp_path)

            self.assertTrue(os.path.exists(self.result_dir))
            self.assertEqual(res, 0)

    @patch('msprobe.pytorch.parse_tool.lib.compare.Compare.convert', return_value=None)
    def test_convert_dump_to_npy(self, mock_convert):
        self.compare.convert_dump_to_npy(self.my_dump_path, None, self.result_dir, self.msaccucmp_path)

        mock_convert.assert_called_once()

    def test_convert(self):
        with patch('msprobe.pytorch.parse_tool.lib.compare.Util.check_path_valid', return_value=True), \
            patch('msprobe.pytorch.parse_tool.lib.compare.Util.check_msaccucmp', return_value=None), \
            patch('msprobe.pytorch.parse_tool.lib.compare.Util.execute_command', return_value=0):
            res = self.compare.convert(self.my_dump_path, None, self.result_dir, self.msaccucmp_path)

            self.assertTrue(os.path.exists(self.result_dir))
            self.assertEqual(res, 0)

    @patch('msprobe.pytorch.parse_tool.lib.compare.Compare.do_compare_data', return_value=(5, False, 0.99493676, 1.0))
    def test_compare_data(self, mock_do_compare_data):
        self.compare.compare_data([self.npy_file1, self.npy_file2, True, 0.001, 0.001, 20])

        self.assertTrue(os.path.exists(self.npy_file1 + '.txt'))
        mock_do_compare_data.assert_called_once()

    def test_do_compare_data(self):
        res = self.compare.do_compare_data(self.var1, self.var2)

        self.assertEqual(res.err, 1.0)

    def test_compare_npy(self):
        self.compare.compare_npy(self.npy_file1, self.npy_file2, self.csv_path)

        self.assertTrue(os.path.exists(self.csv_path))

    @patch('msprobe.pytorch.parse_tool.lib.compare.Compare.compare_npy', return_value=None)
    def test_compare_all_file_in_directory(self, mock_compare_npy):
        with patch('msprobe.pytorch.parse_tool.lib.compare.Util.get_sorted_files_names', return_value=[self.npy_file1, self.npy_file2]), \
            patch('msprobe.pytorch.parse_tool.lib.compare.Util.is_subdir_count_equal', return_value=True), \
            patch('msprobe.pytorch.parse_tool.lib.compare.Util.check_npy_files_valid_in_dir', return_value=True):
            self.compare.compare_all_file_in_directory(self.my_dump_path, self.golden_path, self.result_dir)

            self.assertEqual(mock_compare_npy.call_count, 2)

    @patch('msprobe.pytorch.parse_tool.lib.compare.Compare.compare_all_file_in_directory', return_value=None)
    def test_compare_timestamp_directory(self, mock_compare_all_files_in_directory):
        with patch('msprobe.pytorch.parse_tool.lib.compare.Util.get_sorted_subdirectories_names', return_value=[self.my_dump_path, self.golden_path]), \
            patch('msprobe.pytorch.parse_tool.lib.compare.Util.is_subdir_count_equal', return_value=True):
            self.compare.compare_timestamp_directory(self.my_dump_path, self.golden_path, self.result_dir)

            self.assertEqual(mock_compare_all_files_in_directory.call_count, 2)

    @patch('msprobe.pytorch.parse_tool.lib.compare.Compare.compare_timestamp_directory', return_value=None)
    def test_compare_converted_dir(self, mock_compare_timestamp_directory):
        with patch('msprobe.pytorch.parse_tool.lib.compare.Util.get_sorted_subdirectories_names', return_value=[self.my_dump_path, self.golden_path]), \
            patch('msprobe.pytorch.parse_tool.lib.compare.Util.is_subdir_count_equal', return_value=True), \
            patch('msprobe.pytorch.parse_tool.lib.compare.write_csv', return_value=None), \
            patch('msprobe.pytorch.parse_tool.lib.compare.Util.change_filemode_safe', return_value=None):
            self.compare.compare_converted_dir(self.my_dump_path, self.golden_path, self.result_dir)

            self.assertEqual(mock_compare_timestamp_directory.call_count, 2)

    @patch('msprobe.pytorch.parse_tool.lib.compare.Compare.convert_dump_to_npy', return_value=None)
    def test_convert_api_dir_to_npy(self, mock_convert_dump_to_npy):
        with patch('msprobe.pytorch.parse_tool.lib.compare.Util.check_path_valid', return_value=True):
            self.compare.convert_api_dir_to_npy(self.my_dump_path, None, self.result_dir, self.msaccucmp_path)

            self.assertEqual(mock_convert_dump_to_npy.call_count, 0)
