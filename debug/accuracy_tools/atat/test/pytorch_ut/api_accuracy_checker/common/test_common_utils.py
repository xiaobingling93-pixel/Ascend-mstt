import os
import json
import csv
import unittest
from unittest.mock import patch

from atat.pytorch.api_accuracy_checker.common.utils import *


class TestUtils(unittest.TestCase):

    @patch('atat.pytorch.api_accuracy_checker.common.utils.get_file_content_bytes')
    def test_get_json_contents_should_raise_exception(self, mock_get_file_content_bytes):
        mock_get_file_content_bytes.return_value = 'not a dict'
        with self.assertRaises(CompareException) as ce:
            get_json_contents('')
        self.assertEqual(ce.exception.code, CompareException.INVALID_FILE_ERROR)

    def test_get_json_contents_should_return_json_obj(self):
        test_dict = {"key": "value"}
        file_name = 'test.json'

        fd = os.open(file_name, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, 'w') as f:
            json.dump(test_dict, f)
        self.assertEqual(get_json_contents(file_name), test_dict)
        os.remove(file_name)

    def test_write_csv(self):
        test_file_name = 'test.csv'
        test_data = [["name", "age"], ["Alice", "20"], ["Bob", "30"]]
        write_csv(test_data, 'test.csv')
        with open(test_file_name, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                self.assertEqual(row, test_data[i])
        os.remove(test_file_name)

    def test_check_need_convert(self):
        self.assertEqual(check_need_convert('cross_entropy'), 'int32_to_int64')
        self.assertIsNone(check_need_convert('linear'))

    def test_check_object_type(self):
        try:
            check_object_type(123, int)
        except Exception as e:
            self.fail(f"check_object_type raised exception {e}")

    def test_check_file_or_directory_path(self):
        try:
            check_file_or_directory_path(__file__)
        except Exception as e:
            self.fail(f"check_file_or_directory_path raised exception {e}")

    def test_create_directory(self):
        test_dir_name = 'test_dir'
        create_directory(test_dir_name)
        self.assertTrue(os.path.exists(test_dir_name))
        os.rmdir(test_dir_name)

    def test_get_file_content_bytes(self):
        fd = os.open('test.txt', os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, 'w') as f:
            f.write("Hello, World!")
        self.assertEqual(get_file_content_bytes('test.txt'), b"Hello, World!")
        os.remove('test.txt')

    @patch('os.path.exists')
    def test_check_file_or_dir_path_should_raise_exe_when_dir_path_not_existed(self, mock_path_exists):
        mock_path_exists.return_value = False
        with self.assertRaises(CompareException) as ce:
            check_file_or_directory_path('', isdir=True)
        self.assertEqual(ce.exception.code, CompareException.INVALID_PATH_ERROR)

    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('os.access')
    def test_check_file_or_dir_path_should_pass_when_path_is_dir(self, mock_os_access, mock_path_is_dir,
                                                                 mock_path_exists):
        mock_os_access.return_value = True
        mock_path_is_dir.return_value = True
        mock_path_exists.return_value = True
        check_file_or_directory_path('', isdir=True)

    @patch('os.path.isfile')
    @patch('os.access')
    def test_check_file_or_dir_path_should_raise_exe_when_file_not_access(self, mock_os_access, mock_path_is_file):
        mock_os_access.return_value = False
        mock_path_is_file.return_value = True
        with self.assertRaises(CompareException) as ce:
            check_file_or_directory_path('', isdir=False)
        self.assertEqual(ce.exception.code, CompareException.INVALID_PATH_ERROR)

    def test_check_file_or_dir_path_should_pass_when_path_is_file(self):
        with unittest.mock.patch('os.path.isfile', return_value=True), \
                unittest.mock.patch('os.access', return_value=True):
            check_file_or_directory_path('', isdir=False)

    def test_api_info_preprocess_no_conversion_needed(self):
        api_name = 'linear'
        original_api_info = {'key': 'value'}
        convert_type, processed_api_info = api_info_preprocess(api_name, original_api_info.copy())
        self.assertIsNone(convert_type)
        self.assertEqual(original_api_info, processed_api_info)

    def test_api_info_preprocess_cross_entropy_positive(self):
        api_name = 'cross_entropy'
        api_info = {'args': [{'Name': 'logit'}, {'Name': 'labels', 'Min': 1}]}
        convert_type, processed_api_info = api_info_preprocess(api_name, api_info.copy())
        self.assertEqual(convert_type, 'int32_to_int64')
        self.assertEqual(processed_api_info, api_info)
