
import json
import csv
import unittest
from unittest.mock import patch

from msprobe.pytorch.api_accuracy_checker.common.utils import *
from msprobe.core.common.utils import write_csv

class TestUtils(unittest.TestCase):

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

    def test_create_directory(self):
        test_dir_name = 'test_dir'
        create_directory(test_dir_name)
        self.assertTrue(os.path.exists(test_dir_name))
        os.rmdir(test_dir_name)

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
