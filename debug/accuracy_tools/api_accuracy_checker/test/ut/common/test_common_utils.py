import unittest
import os
import numpy as np
import torch
from api_accuracy_checker.common.utils import *

class TestUtils(unittest.TestCase):

    def test_read_json(self):
        test_dict = {"key": "value"}
        with open('test.json', 'w') as f:
            json.dump(test_dict, f)
        self.assertEqual(read_json('test.json'), test_dict)
        os.remove('test.json')

    def test_write_csv(self):
        test_data = [["name", "age"], ["Alice", "20"], ["Bob", "30"]]
        write_csv(test_data, 'test.csv')
        with open('test.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                self.assertEqual(row, test_data[i])
        os.remove('test.csv')

    def test_print_info_log(self):
        try:
            print_info_log("Test message")
        except Exception as e:
            self.fail(f"print_info_log raised exception {e}")

    def test_check_mode_valid(self):
        try:
            check_mode_valid(Const.ALL)
        except Exception as e:
            self.fail(f"check_mode_valid raised exception {e}")

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

    def test_get_dump_data_path(self):
        path, exist = get_dump_data_path(os.path.dirname(__file__))
        self.assertTrue(exist)

    def test_create_directory(self):
        create_directory('test_dir')
        self.assertTrue(os.path.exists('test_dir'))
        os.rmdir('test_dir')

    def test_execute_command(self):
        execute_command(['echo', 'Hello, World!'])

    def test_parse_arg_value(self):
        values = "1,2,3;4,5,6"
        expected_result = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(parse_arg_value(values), expected_result)

    def test_parse_value_by_comma(self):
        value = "1,2,3"
        expected_result = [1, 2, 3]
        self.assertEqual(parse_value_by_comma(value), expected_result)

    def test_get_data_len_by_shape(self):
        shape = [2, 3, 4]
        expected_result = 24
        self.assertEqual(get_data_len_by_shape(shape), expected_result)

    def test_add_time_as_suffix(self):
        name = "test"
        result = add_time_as_suffix(name)
        self.assertTrue(result.startswith(name))

    def test_get_time(self):
        result = get_time()
        self.assertTrue(isinstance(result, str))

    def test_format_value(self):
        value = 123.456789
        expected_result = '123.456789'
        self.assertEqual(format_value(value), expected_result)

    def test_seed_all(self):
        seed_all(1234)

    def test_get_process_rank(self):
        model = torch.nn.Linear(10, 10)
        rank, _ = get_process_rank(model)
        self.assertEqual(rank, 0)

    def test_get_json_contents(self):
        test_dict = {"key": "value"}
        with open('test.json', 'w') as f:
            json.dump(test_dict, f)
        self.assertEqual(get_json_contents('test.json'), test_dict)
        os.remove('test.json')

    def test_get_file_content_bytes(self):
        with open('test.txt', 'w') as f:
            f.write("Hello, World!")
        self.assertEqual(get_file_content_bytes('test.txt'), b"Hello, World!")
        os.remove('test.txt')

    def test_islink(self):
        self.assertFalse(islink(__file__))

    def test_check_path_length_valid(self):
        self.assertTrue(check_path_length_valid(__file__))

    def test_check_path_pattern_valid(self):
        self.assertIsNone(check_path_pattern_valid(__file__))

    def test_check_input_file_valid(self):
        self.assertIsNone(check_input_file_valid(__file__))

    def test_check_need_convert(self):
        self.assertIsNone(check_need_convert("unknown_api"))
