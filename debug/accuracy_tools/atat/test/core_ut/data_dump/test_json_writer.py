import unittest
from atat.core.data_dump.json_writer import DataWriter

import os
import csv
from atat.core.common.file_check import FileOpen
from atat.core.common import utils
from pathlib import Path
import json

class TestDataWriter(unittest.TestCase):
    def test_write_data_to_csv(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(cur_path, "test.csv")

        if os.path.exists(file_path):
            utils.remove_path(file_path)

        data = {"A":"1", "B":"2", "C":"3"}
        result = data.values()
        header = data.keys()
        DataWriter.write_data_to_csv(result, header, file_path)
        with FileOpen(file_path, "r") as f:
            reader = csv.DictReader(f)
            column_first = [row for row in reader][0]
        self.assertEqual(data, column_first)

        
        
        
        data = {"A":"4", "B":"5", "C":"6"}
        result = data.values()
        header = data.keys()
        DataWriter.write_data_to_csv(result, header, file_path)
        with FileOpen(file_path, "r") as f:
            reader = csv.DictReader(f)
            column_last = [row for row in reader][-1]
        self.assertEqual(data, column_last)

        utils.remove_path(file_path)

    def test_initialize_json_file(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        dump_tensor_data_dir = os.path.join(cur_path, "dump_tensor_data.json")
        dump_file_path = os.path.join(cur_path, "dump_file.json")
        stack_file_path = os.path.join(cur_path, "stack_file.json")
        construct_file_path = os.path.join(cur_path, "construct_file.json")
        if not os.path.exists(stack_file_path):
            Path(stack_file_path).touch()
        if not os.path.exists(construct_file_path):
            Path(construct_file_path).touch()

        test = DataWriter()
        test.stack_file_path = stack_file_path
        test.dump_file_path = dump_file_path
        test.dump_tensor_data_dir = dump_tensor_data_dir
        test.construct_file_path = construct_file_path

        test.initialize_json_file()

        with open(dump_file_path) as f:
            load_data = json.load(f)
        result = {"dump_data_dir": dump_tensor_data_dir, "data": {}}
        self.assertEqual(result, load_data)
        is_exist_1 = os.path.exists(test.stack_file_path)
        self.assertTrue(is_exist_1)
        is_exist_2 = os.path.exists(test.construct_file_path)
        self.assertTrue(is_exist_2)

        os.remove(construct_file_path)
        os.remove(stack_file_path)
        os.remove(dump_file_path)

    def test_update_dump_paths(self):
        test = DataWriter()
        self.assertTrue(test.dump_file_path == None)

        cur_path = os.path.dirname(os.path.realpath(__file__))
        test_path = os.path.join(cur_path, "test1.json")

        test.update_dump_paths(test_path, test_path, test_path, test_path, test_path)
        self.assertTrue(test.dump_file_path == test_path)
        self.assertTrue(test.stack_file_path == test_path)
        self.assertTrue(test.construct_file_path == test_path)
        self.assertTrue(test.dump_tensor_data_dir == test_path)
        self.assertTrue(test.free_benchmark_file_path == test_path)

    def test_update_data(self):
        data = {"A":"1", "B":"2", "C":{"D":"2"}}
        test = DataWriter()
        test.cache_data["data"]["test_1"] = True
        test.cache_data["data"]["test_2"] = False

        test.update_data(data)
        self.assertEqual(test.cache_data["data"]["A"], "1")

        new_data = {"C":{"F":3}}
        test.update_data(new_data)
        self.assertEqual(test.cache_data["data"]["C"]["F"], 3)


    def test_flush_data_when_buffer_is_full_and_test_write_data_json(self):
        data = {"A":"1", "B":"2", "data":{}}
        test = DataWriter()
        test.buffer_size = 1
        test.cache_data["data"] = {"A":"1", "B":"2", "C":"3"}

        self.assertTrue(len(test.cache_data["data"]) >= test.buffer_size)
        cur_path = os.path.dirname(os.path.realpath(__file__))
        dump_tensor_data_dir = os.path.join(cur_path, "dump_tensor_data.json")
        dump_file_path = os.path.join(cur_path, "dump_file.json")
        stack_file_path = os.path.join(cur_path, "stack_file.json")
        construct_file_path = os.path.join(cur_path, "construct_file.json")

        test.dump_file_path = dump_file_path
        test.dump_tensor_data_dir = dump_tensor_data_dir

        with open(dump_file_path, "w") as f:
            dump_data = json.dumps(data)
            f.write(dump_data)

        test.flush_data_when_buffer_is_full()

        with open(dump_file_path, "r") as f:
            new_data = json.load(f)
        
        data.update({"data": {"A":"1", "B":"2", "C":"3"}})
        self.assertEqual(new_data, data)

        self.assertTrue(test.cache_data["data"] == {})
        os.remove(dump_file_path)


    def test_update_stack(self):
        data = {"A":"1", "B":"2", "data":{}}
        test = DataWriter()
        test.update_stack(data)
        self.assertEqual(test.cache_stack, data)

    def test_update_construct(self):
        data = {"A":"1", "B":"2", "data":{}}
        test = DataWriter()
        test.update_construct(data)
        self.assertEqual(test.cache_construct, data)

    def test_write_stack_info_json(self):
        test = DataWriter()
        data = {"A":"1", "B":"2", "data":{}}
        test.cache_stack = data

        cur_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(cur_path, "dump.json")

        test.write_stack_info_json(file_path)

        with open(file_path, "r") as f:
            load_result = json.load(f)
        try:
            self.assertEqual(load_result, data)
        finally:
            os.remove(file_path)


    def test_write_construct_info_json(self):
        test = DataWriter()
        data = {"A":"1", "B":"2", "data":{}}
        test.cache_construct = data

        cur_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(cur_path, "dump.json")

        test.write_construct_info_json(file_path)

        with open(file_path, "r") as f:
            load_result = json.load(f)
        try:
            self.assertEqual(load_result, data)
        finally:
            os.remove(file_path)
