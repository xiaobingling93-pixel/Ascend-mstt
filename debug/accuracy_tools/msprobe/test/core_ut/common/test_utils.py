#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2024-2024. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import json
import os
import tempfile
from unittest import TestCase
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import (FileCheckConst,
                                            FileCheckException,
                                            check_file_size,
                                            check_file_or_directory_path,
                                            get_json_contents,
                                            get_file_content_bytes,
                                            save_json)
from msprobe.core.common.inplace_op_checker import InplaceOpChecker
from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.common.utils import (CompareException,
                                       check_compare_param,
                                       check_configuration_param,
                                       _check_json,
                                       check_json_file,
                                       check_regex_prefix_format_valid,
                                       set_dump_path,
                                       get_dump_mode,
                                       get_real_step_or_rank, 
                                       get_step_or_rank_from_string, 
                                       get_stack_construct_by_dump_json_path,
                                       check_seed_all,
                                       safe_get_value,
                                       MsprobeBaseException,
                                       is_json_file)


class TestUtils(TestCase):

    def test_check_file_or_directory_path(self):
        class TestFileChecker:
            file_path = ""
            path_type = ""
            ability = ""
            checked = False

            def __init__(self, file_path, path_type, ability=None):
                TestFileChecker.file_path = file_path
                TestFileChecker.path_type = path_type
                TestFileChecker.ability = ability

            def common_check(self):
                TestFileChecker.checked = True

        file_path = os.path.realpath(__file__)
        dirname = os.path.dirname(file_path)

        with patch("msprobe.core.common.file_utils.FileChecker", new=TestFileChecker):
            check_file_or_directory_path(file_path, isdir=False)
        self.assertTrue(TestFileChecker.checked)
        self.assertEqual(TestFileChecker.file_path, file_path)
        self.assertEqual(TestFileChecker.path_type, FileCheckConst.FILE)
        self.assertEqual(TestFileChecker.ability, FileCheckConst.READ_ABLE)

        TestFileChecker.checked = False
        with patch("msprobe.core.common.file_utils.FileChecker", new=TestFileChecker):
            check_file_or_directory_path(dirname, isdir=True)
        self.assertTrue(TestFileChecker.checked)
        self.assertEqual(TestFileChecker.file_path, dirname)
        self.assertEqual(TestFileChecker.path_type, FileCheckConst.DIR)
        self.assertEqual(TestFileChecker.ability, FileCheckConst.WRITE_ABLE)

    @patch.object(logger, "error")
    def test_check_compare_param(self, mock_error):
        params = {
            "npu_json_path": "npu_path.json",
            "bench_json_path": "bench_path.json",
            "stack_json_path": "stack_path.json",
            "npu_dump_data_dir": "npu_dump_data_dir",
            "bench_dump_data_dir": "bench_dump_data_dir"
        }

        call_args = [
            ("npu_path.json", False),
            ("bench_path.json", False),
            ("stack_path.json", False),
            ("npu_dump_data_dir", True),
            ("bench_dump_data_dir", True),
            ("output_path", True),
            ("npu_path.json", False),
            ("bench_path.json", False),
            ("stack_path.json", False),
            ("output_path", True)
        ]

        with self.assertRaises(CompareException) as context:
            check_compare_param("npu_path", "output_path", dump_mode=Const.ALL)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with("Invalid input parameter 'input_param', "
                                      "the expected type dict but got <class 'str'>.")

        mock_check_file_or_directory_path = MagicMock()
        mock_check_json_file = MagicMock()
        with patch("msprobe.core.common.utils.FileOpen", mock_open(read_data="")), \
                patch("msprobe.core.common.utils.check_json_file", new=mock_check_json_file), \
                patch("msprobe.core.common.utils.check_file_or_directory_path", new=mock_check_file_or_directory_path):
            check_compare_param(params, "output_path", dump_mode=Const.ALL)
            check_compare_param(params, "output_path", dump_mode=Const.MD5)
        for i in range(len(call_args)):
            self.assertEqual(mock_check_file_or_directory_path.call_args_list[i][0], call_args[i])
        self.assertEqual(len(mock_check_json_file.call_args[0]), 4)
        self.assertEqual(mock_check_json_file.call_args[0][0], params)

    @patch.object(logger, "error")
    def test_check_configuration_param(self, mock_error):
        with self.assertRaises(CompareException) as context:
            check_configuration_param(stack_mode="False", auto_analyze=True, fuzzy_match=False,
                                      is_print_compare_log=True)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with("Invalid input parameter, False which should be only bool type.")

    @patch.object(logger, "error")
    def test__check_json(self, mock_error):
        class TestOpen:
            def __init__(self, string):
                self.string = string

            def readline(self):
                return self.string

            def seek(self, begin, end):
                self.string = str(begin) + "_" + str(end)

        with self.assertRaises(CompareException) as context:
            _check_json(TestOpen(""), "test.json")
        self.assertEqual(context.exception.code, CompareException.INVALID_DUMP_FILE)
        mock_error.assert_called_with("dump file test.json have empty line!")

        handler = TestOpen("jons file\n")
        _check_json(handler, "test.json")
        self.assertEqual(handler.string, "0_0")

    @patch("msprobe.core.common.utils._check_json")
    def test_check_json_file(self, _mock_check_json):
        input_param = {
            "npu_json_path": "npu_path",
            "bench_json_path": "bench_path",
            "stack_json_path": "stack_path"
        }
        check_json_file(input_param, "npu_json", "bench_json", "stack_json")
        self.assertEqual(_mock_check_json.call_args_list[0][0], ("npu_json", "npu_path"))
        self.assertEqual(_mock_check_json.call_args_list[1][0], ("bench_json", "bench_path"))
        self.assertEqual(_mock_check_json.call_args_list[2][0], ("stack_json", "stack_path"))

    @patch.object(logger, "error")
    def test_check_file_size(self, mock_error):
        with patch("msprobe.core.common.utils.os.path.getsize", return_value=120):
            with self.assertRaises(FileCheckException) as context:
                check_file_size("input_file", 100)
        self.assertEqual(context.exception.code, FileCheckException.FILE_TOO_LARGE_ERROR)
        mock_error.assert_called_with("The size (120) of input_file exceeds (100) bytes, tools not support.")

    def test_check_regex_prefix_format_valid(self):
        prefix = "A" * 21
        with self.assertRaises(ValueError) as context:
            check_regex_prefix_format_valid(prefix)
        self.assertEqual(str(context.exception), f"Maximum length of prefix is {Const.REGEX_PREFIX_MAX_LENGTH}, "
                                                 f"while current length is {len(prefix)}")

        prefix = "(prefix)"
        with self.assertRaises(ValueError) as context:
            check_regex_prefix_format_valid(prefix)
        self.assertEqual(str(context.exception), f"prefix contains invalid characters, "
                                                 f"prefix pattern {Const.REGEX_PREFIX_PATTERN}")

    @patch.object(logger, "error")
    def test_set_dump_path(self, mock_error):
        input_param = {
            "npu_json_path": None,
            "bench_json_path": "bench_path"
        }

        with self.assertRaises(CompareException) as context:
            set_dump_path(input_param)
        self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)
        mock_error.assert_called_with("Please check the json path is valid.")

    @patch.object(logger, "error")
    def test_get_dump_mode(self, mock_error):
        input_param = {
            "npu_json_path": None,
            "bench_json_path": "bench_path"
        }
        npu_json = {
            "task": Const.TENSOR,
            "dump_data_dir": "dump_data_dir",
            "data": "data"
        }

        input_param["npu_json_path"] = "npu_path"
        with patch("msprobe.core.common.utils.load_json", return_value=npu_json):
            dump_mode = get_dump_mode(input_param)
        self.assertEqual(dump_mode, Const.ALL)

        npu_json["task"] = Const.STATISTICS
        with patch("msprobe.core.common.utils.load_json", return_value=npu_json), \
                patch("msprobe.core.common.utils.md5_find", return_value=True):
            dump_mode = get_dump_mode(input_param)
        self.assertEqual(dump_mode, Const.MD5)

        npu_json["task"] = Const.OVERFLOW_CHECK
        with patch("msprobe.core.common.utils.load_json", return_value=npu_json):
            with self.assertRaises(CompareException) as context:
                dump_mode = get_dump_mode(input_param)
            self.assertEqual(context.exception.code, CompareException.INVALID_TASK_ERROR)
            mock_error.assert_called_with("Compare applies only to task is tensor or statistics")

    @patch('msprobe.core.common.file_utils.get_file_content_bytes')
    def test_get_json_contents_should_raise_exception(self, mock_get_file_content_bytes):
        mock_get_file_content_bytes.return_value = 'not a dict'
        with self.assertRaises(FileCheckException) as ce:
            get_json_contents('')
        self.assertEqual(ce.exception.code, FileCheckException.INVALID_FILE_ERROR)

    def test_get_json_contents_should_return_json_obj(self):
        test_dict = {"key": "value"}
        file_name = 'test.json'

        fd = os.open(file_name, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, 'w') as f:
            json.dump(test_dict, f)
        self.assertEqual(get_json_contents(file_name), test_dict)
        os.remove(file_name)

    def test_get_file_content_bytes(self):
        fd = os.open('test.txt', os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, 'w') as f:
            f.write("Hello, World!")
        self.assertEqual(get_file_content_bytes('test.txt'), b"Hello, World!")
        os.remove('test.txt')


    def test_get_real_step_or_rank(self):
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank([], "invalid_obj")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        result = get_real_step_or_rank(None, "step")
        self.assertEqual(result, [])
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank("not_a_list", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank([-1, 1, 2], "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank([1, 2, 3.5], "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_real_step_or_rank([True, 1, 2], "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        result = get_real_step_or_rank([1, 10, 50], "step")
        self.assertEqual(result, [1, 10, 50])

    def test_get_step_or_rank_from_string(self):
        with self.assertRaises(MsprobeException) as context:
            get_step_or_rank_from_string("1-4-5", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_step_or_rank_from_string("!-,", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_step_or_rank_from_string("5-3", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            get_step_or_rank_from_string("5-100000000000000", "step")
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        result = get_real_step_or_rank(["1-5", 10], "rank")
        self.assertEqual(result, [1, 2, 3, 4, 5, 10])
        result = get_real_step_or_rank([10, "1-3", 3], "step")
        self.assertEqual(result, [1, 2, 3, 10])

    def test_get_stack_construct_by_dump_json_path_when_dump_json_path_is_none_then_fail(self):
        dump_json_path = None
        with self.assertRaises(CompareException) as context:
            stack, construct = get_stack_construct_by_dump_json_path(dump_json_path)
        self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)

    def test_get_stack_construct_by_dump_json_path_when_dump_json_path_invalid_then_fail(self):
        dump_json_path = "./abc/dump.json"
        framework = Const.MS_FRAMEWORK
        with self.assertRaises(FileCheckException) as context:
            stack, construct = get_stack_construct_by_dump_json_path(dump_json_path)
        self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)

    def test_get_stack_construct_by_dump_json_path_valid_paths_then_pass(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_json_path = os.path.join(temp_dir, 'stack.json')
            construct_json_path = os.path.join(temp_dir, 'construct.json')
            dump_json_path = os.path.join(temp_dir, 'dump.json')

            save_json(stack_json_path, {'stack_key': 'stack_value'})
            save_json(construct_json_path, {'construct_key': 'construct_value'})
            save_json(dump_json_path, {'dump_key': 'dump_value'})

            stack, construct = get_stack_construct_by_dump_json_path(dump_json_path)

            self.assertEqual(stack, {'stack_key': 'stack_value'})
            self.assertEqual(construct, {'construct_key': 'construct_value'})

    def test_check_seed_all(self):
        with self.assertRaises(MsprobeException) as context:
            check_seed_all(-1, True)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            check_seed_all(Const.MAX_SEED_VALUE + 1, True)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            check_seed_all("1", True)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            check_seed_all(True, True)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)
        with self.assertRaises(MsprobeException) as context:
            check_seed_all(True, 1)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_safe_get_value_dict_valid_key_index(self):
        # Test valid key and index in a dictionary
        dict_container = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        self.assertEqual(safe_get_value(dict_container, 1, 'dict_container', key='a'), 2)

    def test_safe_get_value_invalid_key(self):
        # Test invalid key in dictionary
        dict_container = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        with self.assertRaises(MsprobeBaseException) as context:
            safe_get_value(dict_container, 1, 'dict_container', key='invalid_key')
        self.assertEqual(context.exception.code, MsprobeBaseException.INVALID_OBJECT_TYPE_ERROR)

    def test_safe_get_value_valid_key_invalid_index(self):
        # Test invalid index in dictionary[key]
        dict_container = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        with self.assertRaises(MsprobeBaseException) as context:
            safe_get_value(dict_container, 5, 'dict_container', key='a')
        self.assertEqual(context.exception.code, MsprobeBaseException.INDEX_OUT_OF_BOUNDS_ERROR)

    def test_safe_get_value_list_valid_index(self):
        # Test valid index in a list
        list_container = [10, 20, 30]
        self.assertEqual(safe_get_value(list_container, 1, 'list_container'), 20)

    def test_safe_get_value_list_index_out_of_bounds(self):
        # Test index out of bounds in a list
        list_container = [10, 20, 30]
        with self.assertRaises(MsprobeBaseException) as context:
            safe_get_value(list_container, 10, 'list_container')
        self.assertEqual(context.exception.code, MsprobeBaseException.INDEX_OUT_OF_BOUNDS_ERROR)

    def test_safe_get_value_tuple_valid_index(self):
        # Test valid index in a tuple
        tuple_container = (100, 200, 300)
        self.assertEqual(safe_get_value(tuple_container, 2, 'tuple_container'), 300)

    def test_safe_get_value_array_valid_index(self):
        # Test valid index in a numpy array
        array_container = np.array([1000, 2000, 3000])
        self.assertEqual(safe_get_value(array_container, 0, 'array_container'), 1000)

    def test_safe_get_value_unsupported_container_type(self):
        # Test unsupported container type (e.g., a string)
        with self.assertRaises(MsprobeBaseException) as context:
            safe_get_value("unsupported_type", 0, 'string_container')
        self.assertEqual(context.exception.code, MsprobeBaseException.INVALID_OBJECT_TYPE_ERROR)

    def test_is_json_file(self):
        file_path_true = 'step/rank/stack.json'
        file_path_false = 1
        self.assertTrue(is_json_file(file_path_true))
        self.assertFalse(is_json_file(file_path_false))
