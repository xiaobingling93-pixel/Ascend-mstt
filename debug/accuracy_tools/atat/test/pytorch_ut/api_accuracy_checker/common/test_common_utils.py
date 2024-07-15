import unittest
import os
import csv
import json
from atat.pytorch.api_accuracy_checker.common.utils import write_csv, check_object_type, check_file_or_directory_path, \
    create_directory, get_json_contents, get_file_content_bytes, check_need_convert


class TestUtils(unittest.TestCase):

    def test_write_csv(self):
        test_data = [["name", "age"], ["Alice", "20"], ["Bob", "30"]]
        write_csv(test_data, 'test.csv')
        with open('test.csv', 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                self.assertEqual(row, test_data[i])
        os.remove('test.csv')

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
        create_directory('test_dir')
        self.assertTrue(os.path.exists('test_dir'))
        os.rmdir('test_dir')

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

    def test_check_need_convert(self):
        self.assertIsNone(check_need_convert("unknown_api"))
