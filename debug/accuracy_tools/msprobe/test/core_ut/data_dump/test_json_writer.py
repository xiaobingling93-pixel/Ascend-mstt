import csv
import os
import unittest
from unittest.mock import patch

from msprobe.core.common.file_utils import FileOpen, remove_path, load_json
from msprobe.core.data_dump.json_writer import DataWriter


class TestDataWriter(unittest.TestCase):
    def setUp(self):
        self.data_writer = DataWriter()
        self.data_content = {"task": "tensor", "level": "L1", "data": {"Tensor.add": 1}}
        self.cur_path = os.path.dirname(os.path.realpath(__file__))

    def test_write_data_to_csv(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(cur_path, "test.csv")

        if os.path.exists(file_path):
            remove_path(file_path)

        data = {"A": "1", "B": "2", "C": "3"}
        result = data.values()
        header = data.keys()
        DataWriter.write_data_to_csv(result, header, file_path)
        with FileOpen(file_path, "r") as f:
            reader = csv.DictReader(f)
            column_first = [row for row in reader][0]
        self.assertEqual(data, column_first)

        data = {"A": "4", "B": "5", "C": "6"}
        result = data.values()
        header = data.keys()
        DataWriter.write_data_to_csv(result, header, file_path)
        with FileOpen(file_path, "r") as f:
            reader = csv.DictReader(f)
            column_last = [row for row in reader][-1]
        self.assertEqual(data, column_last)

        remove_path(file_path)

    def test_reset_cache(self):
        self.data_writer.cache_data={"data": 1}
        self.data_writer.cache_stack={"stack": 2}
        self.data_writer.cache_construct={"construct": 3}
        self.data_writer.reset_cache()
        self.assertEqual(self.data_writer.cache_data, {})
        self.assertEqual(self.data_writer.cache_stack, {})
        self.assertEqual(self.data_writer.cache_construct, {})

    def test_initialize_json_file(self):
        self.data_writer.dump_tensor_data_dir = "./dump_tensor_data"
        self.data_writer.dump_file_path = os.path.join(self.cur_path, "dump.json")
        self.data_writer.stack_file_path = os.path.join(self.cur_path, "stack.json")
        self.data_writer.construct_file_path = os.path.join(self.cur_path, "construct.json")

        self.data_writer.initialize_json_file(task="tensor", level="L1")
        load_data = load_json(self.data_writer.dump_file_path)
        expected = {"task": "tensor", "level": "L1", "dump_data_dir": "./dump_tensor_data", "data": {}}
        self.assertEqual(load_data, expected)

        expected = {}
        load_data = load_json(self.data_writer.stack_file_path)
        self.assertEqual(load_data, expected)

        load_data = load_json(self.data_writer.construct_file_path)
        self.assertEqual(load_data, expected)

        remove_path(self.data_writer.dump_file_path)
        remove_path(self.data_writer.stack_file_path)
        remove_path(self.data_writer.construct_file_path)

    def test_update_dump_paths(self):
        self.assertIsNone(self.data_writer.dump_file_path)
        test_path = os.path.join(self.cur_path, "test1.json")

        self.data_writer.update_dump_paths(test_path, test_path, test_path, test_path, test_path)
        self.assertTrue(self.data_writer.dump_file_path == test_path)
        self.assertTrue(self.data_writer.stack_file_path == test_path)
        self.assertTrue(self.data_writer.construct_file_path == test_path)
        self.assertTrue(self.data_writer.dump_tensor_data_dir == test_path)
        self.assertTrue(self.data_writer.free_benchmark_file_path == test_path)

    @patch.object(DataWriter, "write_json")
    def test_flush_data_periodically(self, mock_write_json):
        self.data_writer.cache_data["data"] = {"Tensor.add": 1, "Tensor.mul": 2}
        self.data_writer.flush_size = 2
        self.data_writer.flush_data_periodically()
        mock_write_json.assert_called_once()

    def test_update_data(self):
        self.data_writer.cache_data["data"] = {}

        new_data = {"Tensor.mul1": {"input": 1}}
        expected = {"data": {"Tensor.mul1": {"input": 1}}}
        self.data_writer.update_data(new_data)
        self.assertEqual(self.data_writer.cache_data, expected)

        new_data = {"Tensor.mul1": {"output": 2}}
        expected = {"data": {"Tensor.mul1": {"input": 1, "output": 2}}}
        self.data_writer.update_data(new_data)
        self.assertEqual(self.data_writer.cache_data, expected)

        new_data = {"Tensor.mul2": 2, "Tensor.mul3": 3}
        self.data_writer.update_data(new_data)
        self.assertEqual(self.data_writer.cache_data, expected)

    def test_update_stack(self):
        self.data_writer.update_stack(self.data_content)
        self.assertEqual(self.data_writer.cache_stack, self.data_content)

    def test_update_construct(self):
        self.data_writer.update_construct(self.data_content)
        self.assertEqual(self.data_writer.cache_construct, self.data_content)

    def test_write_data_json(self):
        self.data_writer.cache_data = self.data_content
        file_path = os.path.join(self.cur_path, "dump.json")
        self.data_writer.write_data_json(file_path)
        load_result = load_json(file_path)

        try:
            self.assertEqual(load_result, self.data_content)
        finally:
            os.remove(file_path)

    def test_write_stack_info_json(self):
        self.data_writer.cache_stack = self.data_content
        file_path = os.path.join(self.cur_path, "stack.json")
        self.data_writer.write_stack_info_json(file_path)
        load_result = load_json(file_path)

        try:
            self.assertEqual(load_result, self.data_content)
        finally:
            os.remove(file_path)

    def test_write_construct_info_json(self):
        self.data_writer.cache_construct = self.data_content
        file_path = os.path.join(self.cur_path, "construct.json")
        self.data_writer.write_construct_info_json(file_path)
        load_result = load_json(file_path)

        try:
            self.assertEqual(load_result, self.data_content)
        finally:
            os.remove(file_path)
