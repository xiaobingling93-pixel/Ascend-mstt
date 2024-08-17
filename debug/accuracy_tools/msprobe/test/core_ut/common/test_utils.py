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
import os
import uuid

from unittest import TestCase
from unittest.mock import patch, MagicMock, mock_open

from msprobe.core.common.log import logger
from msprobe.core.common.const import Const
from msprobe.core.common.utils import (CompareException,
                                       check_seed_all,
                                       check_inplace_op,
                                       make_dump_path_if_not_exists,
                                       check_mode_valid,
                                       check_switch_valid,
                                       check_dump_mode_valid,
                                       check_summary_mode_valid,
                                       check_summary_only_valid,
                                       check_file_or_directory_path,
                                       check_compare_param,
                                       check_configuration_param,
                                       is_starts_with,
                                       _check_json,
                                       check_json_file,
                                       check_file_size,
                                       check_regex_prefix_format_valid,
                                       get_dump_data_path,
                                       task_dumppath_get)

from msprobe.core.common.file_check import FileCheckConst
from msprobe.pytorch.common.utils import get_json_contents, get_file_content_bytes

class TestUtils(TestCase):
    @patch.object(logger, "error")
    def test_check_seed_all(self, mock_error):
        self.assertIsNone(check_seed_all(1234, True))
        self.assertIsNone(check_seed_all(0, True))
        self.assertIsNone(check_seed_all(Const.MAX_SEED_VALUE, True))

        with self.assertRaises(CompareException) as context:
            check_seed_all(-1, True)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")

        with self.assertRaises(CompareException) as context:
            check_seed_all(Const.MAX_SEED_VALUE + 1, True)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with(f"Seed must be between 0 and {Const.MAX_SEED_VALUE}.")

        with self.assertRaises(CompareException) as context:
            check_seed_all("1234", True)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with("Seed must be integer.")

        with self.assertRaises(CompareException) as context:
            check_seed_all(1234, 1)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with("seed_all mode must be bool.")

    def test_check_inplace_op(self):
        test_prefix_1 = "Distributed.broadcast.0.forward.input.0"
        self.assertTrue(check_inplace_op(test_prefix_1))
        test_prefix_2 = "Distributed_broadcast_0_forward_input_0"
        self.assertFalse(check_inplace_op(test_prefix_2))
        test_prefix_3 = "Torch.sum.0.backward.output.0"
        self.assertFalse(check_inplace_op(test_prefix_3))

    @patch.object(logger, "error")
    def test_make_dump_path_if_not_exists(self, mock_error):
        file_path = os.path.realpath(__file__)
        dirname = os.path.dirname(file_path) + str(uuid.uuid4())

        def test_mkdir(self, **kwargs):
            raise OSError

        if not os.path.exists(dirname):
            with patch("msprobe.core.common.utils.Path.mkdir", new=test_mkdir):
                with self.assertRaises(CompareException) as context:
                    make_dump_path_if_not_exists(dirname)
                self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)

        make_dump_path_if_not_exists(file_path)
        mock_error.assert_called_with(f"{file_path} already exists and is not a directory.")

    def test_check_mode_valid(self):
        with self.assertRaises(ValueError) as context:
            check_mode_valid("all", scope="scope")
        self.assertEqual(str(context.exception), "scope param set invalid, it's must be a list.")

        with self.assertRaises(ValueError) as context:
            check_mode_valid("all", api_list="api_list")
        self.assertEqual(str(context.exception), "api_list param set invalid, it's must be a list.")

        mode = "all_list"
        with self.assertRaises(CompareException) as context:
            check_mode_valid(mode)
        self.assertEqual(context.exception.code, CompareException.INVALID_DUMP_MODE)
        self.assertEqual(str(context.exception),
                         f"Current mode '{mode}' is not supported. Please use the field in {Const.DUMP_MODE}")

        mode = "list"
        with self.assertRaises(ValueError) as context:
            check_mode_valid(mode)
        self.assertEqual(str(context.exception),
                         "set_dump_switch, scope param set invalid, it's should not be an empty list.")

    @patch.object(logger, "error")
    def test_check_switch_valid(self, mock_error):
        with self.assertRaises(CompareException) as context:
            check_switch_valid("Close")
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with("Please set switch with 'ON' or 'OFF'.")

    @patch.object(logger, "warning")
    def test_check_dump_mode_valid(self, mock_warning):
        dump_mode = check_dump_mode_valid("all")
        mock_warning.assert_called_with("Please set dump_mode as a list.")
        self.assertEqual(dump_mode, ["forward", "backward", "input", "output"])

        with self.assertRaises(ValueError) as context:
            check_dump_mode_valid("all_forward")
        self.assertEqual(str(context.exception),
                         "Please set dump_mode as a list containing one or more of the following: " +
                         "'all', 'forward', 'backward', 'input', 'output'.")

    def test_check_summary_mode_valid(self):
        with self.assertRaises(CompareException) as context:
            check_summary_mode_valid("MD5")
        self.assertEqual(context.exception.code, CompareException.INVALID_SUMMARY_MODE)
        self.assertEqual(str(context.exception), "The summary_mode is not valid")

    @patch.object(logger, "error")
    def test_check_summary_only_valid(self, mock_error):
        summary_only = check_summary_only_valid(True)
        self.assertTrue(summary_only)

        with self.assertRaises(CompareException) as context:
            check_summary_only_valid("True")
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with("Params summary_only only support True or False.")

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

        with patch("msprobe.core.common.utils.FileChecker", new=TestFileChecker):
            check_file_or_directory_path(file_path, isdir=False)
        self.assertTrue(TestFileChecker.checked)
        self.assertEqual(TestFileChecker.file_path, file_path)
        self.assertEqual(TestFileChecker.path_type, FileCheckConst.FILE)
        self.assertEqual(TestFileChecker.ability, FileCheckConst.READ_ABLE)

        TestFileChecker.checked = False
        with patch("msprobe.core.common.utils.FileChecker", new=TestFileChecker):
            check_file_or_directory_path(dirname, isdir=True)
        self.assertTrue(TestFileChecker.checked)
        self.assertEqual(TestFileChecker.file_path, dirname)
        self.assertEqual(TestFileChecker.path_type, FileCheckConst.DIR)
        self.assertEqual(TestFileChecker.ability, FileCheckConst.WRITE_ABLE)

    @patch.object(logger, "error")
    def test_check_compare_param(self, mock_error):
        params = {
            "npu_path": "npu_path",
            "bench_path": "bench_path",
            "stack_path": "stack_path",
            "npu_dump_data_dir": "npu_dump_data_dir",
            "bench_dump_data_dir": "bench_dump_data_dir"
        }

        call_args = [
            ("npu_path", False),
            ("bench_path", False),
            ("stack_path", False),
            ("npu_dump_data_dir", True),
            ("bench_dump_data_dir", True),
            ("output_path", True),
            ("npu_path", False),
            ("bench_path", False),
            ("stack_path", False),
            ("output_path", True)
        ]

        with self.assertRaises(CompareException) as context:
            check_compare_param("npu_path", "output_path")
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with("Invalid input parameters")

        mock_check_file_or_directory_path = MagicMock()
        mock_check_json_file = MagicMock()
        with patch("msprobe.core.common.utils.FileOpen", mock_open(read_data="")), \
             patch("msprobe.core.common.utils.check_json_file", new=mock_check_json_file), \
             patch("msprobe.core.common.utils.check_file_or_directory_path", new=mock_check_file_or_directory_path):
            check_compare_param(params, "output_path")
            check_compare_param(params, "output_path", summary_compare=False, md5_compare=True)
        for i in range(len(call_args)):
            self.assertEqual(mock_check_file_or_directory_path.call_args_list[i][0], call_args[i])
        self.assertEqual(len(mock_check_json_file.call_args[0]), 4)
        self.assertEqual(mock_check_json_file.call_args[0][0], params)

    @patch.object(logger, "error")
    def test_check_configuration_param(self, mock_error):
        with self.assertRaises(CompareException) as context:
            check_configuration_param(stack_mode="False", auto_analyze=True, fuzzy_match=False)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)
        mock_error.assert_called_with("Invalid input parameters which should be only bool type.")

    def test_is_starts_with(self):
        string = "input_slot0"
        self.assertFalse(is_starts_with(string, []))
        self.assertFalse(is_starts_with("", ["input"]))
        self.assertFalse(is_starts_with(string, ["output"]))
        self.assertTrue(is_starts_with(string, ["input", "output"]))

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
            "npu_path": "npu_path",
            "bench_path": "bench_path",
            "stack_path": "stack_path"
        }
        check_json_file(input_param, "npu_json", "bench_json", "stack_json")
        self.assertEqual(_mock_check_json.call_args_list[0][0], ("npu_json", "npu_path"))
        self.assertEqual(_mock_check_json.call_args_list[1][0], ("bench_json", "bench_path"))
        self.assertEqual(_mock_check_json.call_args_list[2][0], ("stack_json", "stack_path"))

    @patch.object(logger, "error")
    def test_check_file_size(self, mock_error):
        with patch("msprobe.core.common.utils.os.path.getsize", return_value=120):
            with self.assertRaises(CompareException) as context:
                check_file_size("input_file", 100)
        self.assertEqual(context.exception.code, CompareException.INVALID_FILE_ERROR)
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

    @patch("msprobe.core.common.utils.check_file_or_directory_path")
    def test_get_dump_data_path(self, mock_check_file_or_directory_path):
        file_path = os.path.realpath(__file__)
        dirname = os.path.dirname(file_path)

        dump_data_path, file_is_exist = get_dump_data_path(dirname)
        self.assertEqual(mock_check_file_or_directory_path.call_args[0], (dirname, True))
        self.assertEqual(dump_data_path, dirname)
        self.assertTrue(file_is_exist)

    @patch.object(logger, "error")
    def test_task_dumppath_get(self, mock_error):
        input_param = {
            "npu_path": None,
            "bench_path": "bench_path"
        }
        npu_json = {
            "task": Const.TENSOR,
            "dump_data_dir": "dump_data_dir",
            "data": "data"
        }

        with self.assertRaises(CompareException) as context:
            task_dumppath_get(input_param)
        self.assertEqual(context.exception.code, CompareException.INVALID_PATH_ERROR)
        mock_error.assert_called_with("Please check the json path is valid.")

        input_param["npu_path"] = "npu_path"
        with patch("msprobe.core.common.utils.FileOpen", mock_open(read_data="")), \
             patch("msprobe.core.common.utils.json.load", return_value=npu_json):
            summary_compare, md5_compare = task_dumppath_get(input_param)
        self.assertFalse(summary_compare)
        self.assertFalse(md5_compare)

        npu_json["task"] = Const.STATISTICS
        with patch("msprobe.core.common.utils.FileOpen", mock_open(read_data="")), \
             patch("msprobe.core.common.utils.json.load", return_value=npu_json), \
             patch("msprobe.core.common.utils.md5_find", return_value=True):
            summary_compare, md5_compare = task_dumppath_get(input_param)
        self.assertFalse(summary_compare)
        self.assertTrue(md5_compare)

        npu_json["task"] = Const.OVERFLOW_CHECK
        with patch("msprobe.core.common.utils.FileOpen", mock_open(read_data="")), \
             patch("msprobe.core.common.utils.json.load", return_value=npu_json):
            with self.assertRaises(CompareException) as context:
                task_dumppath_get(input_param)
            self.assertEqual(context.exception.code, CompareException.INVALID_TASK_ERROR)
            mock_error.assert_called_with("Compare is not required for overflow_check or free_benchmark.")

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

    def test_get_file_content_bytes(self):
        fd = os.open('test.txt', os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        with os.fdopen(fd, 'w') as f:
            f.write("Hello, World!")
        self.assertEqual(get_file_content_bytes('test.txt'), b"Hello, World!")
        os.remove('test.txt')
