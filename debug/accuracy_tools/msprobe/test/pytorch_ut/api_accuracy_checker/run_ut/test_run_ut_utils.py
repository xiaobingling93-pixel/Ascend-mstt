# coding=utf-8
import unittest
from unittest.mock import patch, MagicMock
import torch
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import *


class TestRunUtUtils(unittest.TestCase):

    def test_get_validated_result_csv_patht_valid_mode(self):
        result_csv_path = '/path/to/result.csv'
        validated_path = get_validated_result_csv_path(result_csv_path, 'result')
        self.assertEqual(validated_path, result_csv_path)

    def test_get_validated_result_csv_path_invalid_mode(self):
        with self.assertRaises(ValueError):
            get_validated_result_csv_path('/path/to/result.csv', 'invalid_mode')

    def test_get_validated_result_csv_path_file_path_validation(self):
        with patch('FileChecker') as mock_FileChecker:
            mock_instance = mock_FileChecker.return_value
            mock_instance.common_check.return_value = '/path/to/validated.csv'
            validated_path = get_validated_result_csv_path('/path/to/result.csv', 'result')
            self.assertEqual(validated_path, '/path/to/validated.csv')

    def test_get_validated_result_csv_patht_result_csv_name_pattern(self):
        with patch('your_module.FileChecker') as mock_FileChecker:
            mock_instance = mock_FileChecker.return_value
            mock_instance.common_check.return_value = '/path/to/accuracy_checking_result_12345678901234.csv'
            validated_path = get_validated_result_csv_path('/path/to/result.csv', 'result')
            self.assertEqual(validated_path, '/path/to/validated.csv')

    def test_get_validated_result_csv_path_invalid_result_csv_name_pattern(self):
        with patch('FileChecker') as mock_FileChecker:
            mock_instance = mock_FileChecker.return_value
            mock_instance.common_check.return_value = '/path/to/invalid_name.csv'
            with self.assertRaises(ValueError):
                get_validated_result_csv_path('/path/to/result.csv', 'result')
                
    def test_get_validated_details_csv_path_valid_result_csv_path(self):
        result_csv_path = '/path/to/accuracy_checking_result_12345678901234.csv'
        expected_details_csv_path = '/path/to/accuracy_checking_details_12345678901234.csv'
        with patch('FileChecker') as mock_FileChecker:
            mock_instance = mock_FileChecker.return_value
            mock_instance.common_check.return_value = expected_details_csv_path
            validated_details_csv_path = get_validated_details_csv_path(result_csv_path)
            self.assertEqual(validated_details_csv_path, expected_details_csv_path)

    def test_get_validated_details_csv_path_file_name_replacement(self):
        result_csv_path = '/path/to/result_file.csv'
        expected_details_csv_path = '/path/to/details_file.csv'
        with patch('FileChecker') as mock_FileChecker:
            mock_instance = mock_FileChecker.return_value
            mock_instance.common_check.return_value = expected_details_csv_path
            validated_details_csv_path = get_validated_details_csv_path(result_csv_path)
            self.assertEqual(validated_details_csv_path, expected_details_csv_path)

    def test_get_validated_details_csv_path_file_path_validation(self):
        result_csv_path = '/path/to/result_file.csv'
        with patch('FileChecker') as mock_FileChecker:
            mock_instance = mock_FileChecker.return_value
            mock_instance.common_check.side_effect = Exception("File path is invalid")
            with self.assertRaises(ValueError):
                get_validated_details_csv_path(result_csv_path)
                
    def test_exec_api_functional_api(self):
        api_name = "add"
        args = (torch.tensor(1), torch.tensor(2))
        result = exec_api("Functional", api_name, None, args, kwargs=None)
        self.assertEqual(result, torch.tensor(3))

    def test_exec_api_tensor_api(self):
        api_name = "add"
        args = (torch.tensor(1), torch.tensor(2))
        result = exec_api("Tensor", api_name, None, args, kwargs=None)
        self.assertEqual(result, torch.tensor(3))

    def test_exec_api_torch_api(self):
        api_name = "add"
        args = (torch.tensor(1), torch.tensor(2))
        result = exec_api("Torch", api_name, None, args, kwargs=None)
        self.assertEqual(result, torch.tensor(3))

    def test_exec_api_aten_api(self):
        api_name = "add"
        args = (torch.tensor(1), torch.tensor(2))
        result = exec_api("Aten", api_name, None, args, kwargs=None)
        self.assertEqual(result, torch.tensor(3))

    def test_raise_bench_data_dtype_dtype_unchanged(self):
        arg = torch.tensor(1.0, dtype=torch.float32)
        raise_dtype = torch.float32
        result = raise_bench_data_dtype("api_name", arg, raise_dtype)
        self.assertEqual(result, arg)

    def test_raise_bench_data_dtype_dtype_changed(self):
        arg = torch.tensor(1.0, dtype=torch.float32)
        raise_dtype = torch.float64
        result = raise_bench_data_dtype("api_name", arg, raise_dtype)
        self.assertEqual(result.dtype, raise_dtype)

    def test_raise_bench_data_dtype_hf_32_standard_api(self):
        arg = torch.tensor(1.0, dtype=torch.float32)
        result = raise_bench_data_dtype("conv2d", arg)
        self.assertEqual(result, arg)
