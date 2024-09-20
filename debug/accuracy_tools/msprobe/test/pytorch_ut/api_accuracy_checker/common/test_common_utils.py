
import json
import csv
import unittest
from unittest.mock import patch

from msprobe.pytorch.api_accuracy_checker.common.utils import *
from msprobe.core.common.file_utils import write_csv


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
        
    def test_extract_basic_api_segments(self):
        api_full_name = 'torch.matmul.0'
        api_type, api_name = extract_basic_api_segments(api_full_name)
        self.assertEqual(api_type, 'torch')
        self.assertEqual(api_name, 'matmul')
        
        api_full_name = 'torch.linalg.vector_norm.0'
        api_type, api_name = extract_basic_api_segments(api_full_name)
        self.assertEqual(api_type, 'torch')
        self.assertEqual(api_name, 'linalg.vector_norm')
        
        api_full_name = 'torch.nn.linalg.vector_norm.0'
        api_type, api_name = extract_basic_api_segments(api_full_name)
        self.assertEqual(api_type, None)
        self.assertEqual(api_name, None)
        
    def test_extract_detailed_api_segments(self):
        api_full_name = 'torch.matmul.0.forward.output.0'
        api_name, full_api_name, direction_status = extract_detailed_api_segments(api_full_name)
        self.assertEqual(api_name, 'matmul')
        self.assertEqual(full_api_name, 'torch.matmul.0')
        self.assertEqual(direction_status, 'forward')
        
        api_full_name = 'torch.linalg.vector_norm.0.backward.output.0'
        api_name, full_api_name, direction_status = extract_detailed_api_segments(api_full_name)
        self.assertEqual(api_name, 'linalg.vector_norm')
        self.assertEqual(full_api_name, 'torch.linalg.vector_norm.0')
        self.assertEqual(direction_status, 'backward')
        
        api_full_name = 'torch.nn.functional.linear.0.input.0.1'
        api_name, full_api_name, direction_status = extract_detailed_api_segments(api_full_name)
        self.assertEqual(api_name, None)
        self.assertEqual(full_api_name, None)
        self.assertEqual(direction_status, None)
        
    def test_get_module_and_atttribute_name(self):
        attribute = 'torch.float32'
        module_name, attribute_name = get_module_and_atttribute_name(attribute)
        self.assertEqual(module_name, 'torch')
        self.assertEqual(attribute_name, 'float32')
        
        attribute = 'torch'
        module_name, attribute_name = get_module_and_atttribute_name(attribute)
        with self.assertRaises(CompareException) as context:
            get_module_and_atttribute_name(attribute)
        self.assertTrue(isinstance(context.exception, CompareException))
        self.assertEqual(context.exception.error_code, CompareException.INVALID_DATA_ERROR)
        
    def test_get_attribute(self):
        module_name = 'torch'
        attribute_name = 'float32'
        attribute = get_attribute(module_name, attribute_name)
        self.assertEqual(attribute, torch.float32)

        module_name = 'json'
        attribute_name = 'loads'
        with self.assertRaises(CompareException) as context:
            get_attribute(module_name, attribute_name)
        self.assertTrue(isinstance(context.exception, CompareException))
        self.assertEqual(context.exception.error_code, CompareException.INVALID_DATA_ERROR)
        
        module_name = "torch"
        attribute_name = "float128"
        with self.assertRaises(AttributeError):
            get_attribute(module_name, attribute_name)
        