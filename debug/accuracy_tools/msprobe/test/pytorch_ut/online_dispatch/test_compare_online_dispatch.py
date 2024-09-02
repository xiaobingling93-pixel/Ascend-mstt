# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
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

import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from msprobe.core.common.file_check import FileOpen
from msprobe.core.common.utils import CompareException
from msprobe.pytorch.online_dispatch.compare import get_json_contents, Saver


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.dict_json_path = "./dict.json"
        self.list_json_path = "./list.json"
        Path(self.dict_json_path).touch()
        Path(self.list_json_path).touch()

    def tearDown(self):
        if os.path.exists(self.dict_json_path):
            os.remove(self.dict_json_path)
        if os.path.exists(self.list_json_path):
            os.remove(self.list_json_path)

    def test_get_json_contents_when_get_json(self):
        data = {"one": 1}
        with FileOpen(self.dict_json_path, 'w') as f:
            json.dump(data, f)
        self.assertEqual(get_json_contents(self.dict_json_path), data)

    @patch('msprobe.core.common.log.BaseLogger.error')
    def test_get_json_contents_when_get_list(self, mock_error):
        data = [1, 2]
        with FileOpen(self.list_json_path, 'w') as f:
            json.dump(data, f)
        with self.assertRaises(CompareException) as context:
            get_json_contents(self.list_json_path)
            self.assertEqual(context.exception.code, CompareException.INVALID_FILE_ERROR)
        mock_error.assert_called_once_with('Json file %s, content is not a dictionary!' % self.list_json_path)


class TestSaver(unittest.TestCase):
    def setUp(self):
        self.save_path = "./saver_save.csv"
        self.detail_save_path = "./saver_detail.csv"
        self.saver = Saver(self.save_path, self.detail_save_path, False)
        Path(self.save_path).touch()
        Path(self.detail_save_path).touch()

    def tearDown(self):
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        if os.path.exists(self.detail_save_path):
            os.remove(self.detail_save_path)

    def test_write_csv_title(self):
        self.saver.write_csv_title()
        mock_data_save = {self.saver.COLUMN_API_NAME: {},
                          self.saver.COLUMN_FORWARD_SUCCESS: {},
                          self.saver.COLUMN_BACKWARD_SUCCESS: {},
                          "Message": {}}
        mock_data_detail = {'Npu Name': {}, 'Bench Dtype': {}, 'NPU Dtype': {}, 'Shape': {}, 'error_balance': {},
                            'max_abs_diff': {}, 'max_abs_idx': {}, 'max_rel_diff': {}, 'max_rel_idx': {}, 'eb_thd': {},
                            'error_thd': {}, 'Status': {}, 'Message': {}}
        self.assertEqual(pd.read_csv(self.save_path).to_dict(), mock_data_save)
        self.assertEqual(pd.read_csv(self.detail_save_path).to_dict(), mock_data_detail)

    def test_write_summary_csv(self):
        mock_test_result = Mock()
        mock_test_result.api_name = "api_name"
        mock_test_result.is_fwd_success = "SKIP"
        mock_test_result.is_bwd_success = "NOSKIP"
        mock_test_result.fwd_compare_alg_results = "result"
        self.saver.write_summary_csv(mock_test_result)
        mock_data_save = {"api_name": {}, "SKIP": {}, "NOSKIP": {}, "result": {}}
        self.assertTrue(pd.read_csv(self.save_path).to_dict() == mock_data_save)

    def test_write_detail_csv(self):
        mock_test_result = Mock()
        mock_test_result.api_name = "api_name"
        mock_test_result.fwd_compare_alg_results = ["f"]
        mock_test_result.bwd_compare_alg_results = ["b"]
        self.saver.write_detail_csv(mock_test_result)
        mock_data_detail = {'api_name.forward.output.0': {0: 'api_name.backward.output.0'}, 'f': {0: 'b'}}

        self.assertTrue(pd.read_csv(self.detail_save_path).to_dict() == mock_data_detail)
