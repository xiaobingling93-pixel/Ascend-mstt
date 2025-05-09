import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

from msprobe.core.config_check.checkers.random_checker import compare_json_files, compare_random, get_file_and_line


class TestCompareRandom(unittest.TestCase):

    @patch('os.listdir', return_value=['rank1.json', 'rank2.json'])
    @patch('os.path.join', return_value='test_path')
    @patch("msprobe.core.config_check.checkers.random_checker.load_json")
    def test_compare_random_with_files(self, mock_load_json, mock_path, mock_listdir):
        mock_load_json.return_value = {"op1": {"position1": 1}}
        bench_dir = 'test_bench'
        cmp_dir = 'test_cmp'
        result = compare_random(bench_dir, cmp_dir)
        self.assertEqual(isinstance(result, pd.DataFrame), True)

    @patch('os.listdir', return_value=[])
    @patch('os.path.join', return_value='test_path')
    def test_compare_random_no_files(self, mock_path, mock_listdir):
        bench_dir = 'test_bench'
        cmp_dir = 'test_cmp'
        result = compare_random(bench_dir, cmp_dir)
        self.assertEqual(isinstance(result, pd.DataFrame), True)
        self.assertEqual(len(result), 0)

    def test_get_file_and_line_with_valid_input(self):
        position = '/path/to/file.py:10'
        result = get_file_and_line(position)
        self.assertEqual(isinstance(result, str), True)
        self.assertEqual(result, 'file.py:10')

    def test_get_file_and_line_with_invalid_input(self):
        position = 'invalid_position'
        result = get_file_and_line(position)
        self.assertEqual(isinstance(result, str), True)
        self.assertEqual(result, 'invalid_position')

    @patch('os.listdir', return_value=['rank1.json', 'rank2.json'])
    @patch('os.path.join', return_value='test_path')
    def test_compare_json_files_same_data(self, mock_path, mock_listdir):
        bench_data = {"op1": {"position1:10": 1}}
        cmp_data = {"op1": {"position1:10": 1}}
        result = compare_json_files(bench_data, cmp_data)
        self.assertEqual(isinstance(result, list), True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], True)

    @patch('os.listdir', return_value=['rank1.json', 'rank2.json'])
    @patch('os.path.join', return_value='test_path')
    def test_compare_json_files_different_data(self, mock_path, mock_listdir):
        bench_data = {"op1": {"position1:10": 1}}
        cmp_data = {"op1": {"position1:10": 2}}
        result = compare_json_files(bench_data, cmp_data)
        self.assertEqual(isinstance(result, list), True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], False)

    @patch('os.listdir', return_value=['rank1.json', 'rank2.json'])
    @patch('os.path.join', return_value='test_path')
    def test_compare_json_files_missing_op_in_bench(self, mock_path, mock_listdir):
        bench_data = {}
        cmp_data = {"op1": {"position1:10": 1}}
        result = compare_json_files(bench_data, cmp_data)
        self.assertEqual(isinstance(result, list), True)
        self.assertEqual(len(result), 1)

    @patch('os.listdir', return_value=['rank1.json', 'rank2.json'])
    @patch('os.path.join', return_value='test_path')
    def test_compare_json_files_missing_op_in_cmp(self, mock_path, mock_listdir):
        bench_data = {"op1": {"position1:10": 1}}
        cmp_data = {}
        result = compare_json_files(bench_data, cmp_data)
        self.assertEqual(isinstance(result, list), True)
        self.assertEqual(len(result), 1)
