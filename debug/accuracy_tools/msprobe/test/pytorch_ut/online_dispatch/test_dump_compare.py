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

import unittest
import torch
import json
import os
import threading
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from msprobe.core.common.const import CompareConst
from msprobe.pytorch.online_dispatch.dump_compare import support_basic_type, dump_data, save_temp_summary, dispatch_workflow, get_torch_func, dispatch_multiprocess, error_call, save_csv



class TestDumpCompare(unittest.TestCase):
    def setUp(self):
        self.summary_path = "summary.json"
        Path(self.summary_path).touch()
        self.csv_path = "test_save_csv.csv"
        Path(self.csv_path).touch()
        self.data = {CompareConst.NPU_NAME: 1,
                    CompareConst.BENCH_NAME: 1,
                    CompareConst.NPU_DTYPE: 1,
                    CompareConst.BENCH_DTYPE: 11,
                    CompareConst.NPU_SHAPE: 1,
                    CompareConst.BENCH_SHAPE: 1,
                    CompareConst.NPU_MAX: 1,
                    CompareConst.NPU_MIN: 1,
                    CompareConst.NPU_MEAN: 1,
                    CompareConst.BENCH_MAX: 1,
                    CompareConst.BENCH_MIN: 1,
                    CompareConst.BENCH_MEAN: 1,
                    CompareConst.COSINE: 1,
                    CompareConst.MAX_ABS_ERR: 1,
                    CompareConst.MAX_RELATIVE_ERR: 1,
                    CompareConst.ACCURACY: 1,
                    CompareConst.ERROR_MESSAGE: 1}
        self.data_gt = {CompareConst.NPU_NAME: 1,
                    CompareConst.BENCH_NAME: 1,
                    CompareConst.NPU_DTYPE: 1,
                    CompareConst.BENCH_DTYPE: 11,
                    CompareConst.NPU_SHAPE: 1,
                    CompareConst.BENCH_SHAPE: 1,
                    CompareConst.NPU_MAX: 1,
                    CompareConst.NPU_MIN: 1,
                    CompareConst.NPU_MEAN: 1,
                    CompareConst.BENCH_MAX: 1,
                    CompareConst.BENCH_MIN: 1,
                    CompareConst.BENCH_MEAN: 1,
                    CompareConst.COSINE: 1,
                    CompareConst.MAX_ABS_ERR: 1,
                    CompareConst.MAX_RELATIVE_ERR: 1,
                    CompareConst.ACCURACY: 1,
                    CompareConst.STACK: 2,
                    CompareConst.ERROR_MESSAGE: 1}

    def tearDown(self):
        if os.path.exists(self.summary_path):
            os.remove(self.summary_path)
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_support_basic_type_should_return_true_when_is_instance(self):
        self.assertTrue(support_basic_type(2.3))

    def test_support_basic_type_should_return_false_when_isnot_instance(self):
        self.assertFalse(support_basic_type("abcde"))

    def test_save_temp_summary(self):
        api_index='1'
        single_api_summary="conv2d"
        path = ''
        data = []
        lock=threading.Lock()

        save_temp_summary(api_index=api_index,single_api_summary=single_api_summary,path=path,lock=lock)

        with open(self.summary_path, 'r') as f:
                content = f.readlines() 
                for line in content:
                    data.append(json.loads(line))
        self.assertEqual([['1','conv2d']],data)

    @patch('msprobe.pytorch.online_dispatch.dump_compare.dump_data')
    @patch('msprobe.pytorch.online_dispatch.dump_compare.save_temp_summary')
    def test_dispatch_workflow_should_dump_when_flag_is_True(self,mock_save_temp_summary,mock_dump_data):
        mock_run_param = Mock()
        mock_run_param.aten_api="aten_api"
        mock_run_param.single_api_index="single_api_index"
        mock_run_param.root_npu_path=""
        mock_data_info = Mock()
        mock_data_info.cpu_args=None
        mock_data_info.cpu_kwargs=[]

        mock_run_param.dump_flag=True
        mock_run_param.process_num = 0  
        mock_run_param.api_index = 1
        mock_data_info.all_summary=[1]

        dispatch_workflow(mock_run_param, mock_data_info)
        mock_dump_data.assert_called()
        mock_save_temp_summary.assert_not_called()

    @patch('msprobe.pytorch.online_dispatch.dump_compare.dump_data')
    @patch('msprobe.pytorch.online_dispatch.dump_compare.save_temp_summary')
    def test_dispatch_workflow_should_not_dump_when_flag_is_false(self,mock_save_temp_summary,mock_dump_data):
        mock_run_param = Mock()
        mock_run_param.aten_api="aten_api"
        mock_run_param.single_api_index="single_api_index"
        mock_run_param.root_npu_path=""
        mock_data_info = Mock()
        mock_data_info.cpu_args=None
        mock_data_info.cpu_kwargs=[]

        mock_run_param.dump_flag=False
        mock_run_param.auto_dump_flag=False
        mock_run_param.process_num = 1
        mock_run_param.api_index = 1
        mock_data_info.all_summary=[1]

        dispatch_workflow(mock_run_param, mock_data_info)
        mock_dump_data.assert_not_called()
        mock_save_temp_summary.assert_called()   

    def test_get_torch_func_should_return_None_when_outside_input(self):
        mock_run_param = Mock()
        mock_run_param.func_namespace="new_attr1"
        mock_run_param.aten_api="new_attr2"
        mock_run_param.aten_api_overload_name="new_attr3"
        self.assertIsNone(get_torch_func(mock_run_param))

    def test_get_torch_func_should_return_None_when_inside_input(self):
        mock_run_param = Mock()
        mock_run_param.func_namespace="aten"
        mock_run_param.aten_api="add"
        mock_run_param.aten_api_overload_name="Scalar"
        self.assertEqual(get_torch_func(mock_run_param),torch.ops.aten.add.Scalar)

    @patch('msprobe.core.common.log.BaseLogger.error')
    def test_dispatch_multiprocess_should_logger_error_when_wrong_api_input(self,mock_error):
        mock_run_param = Mock()
        mock_run_param.func_namespace="new_attr1"
        mock_run_param.aten_api="new_attr2"
        mock_run_param.aten_api_overload_name="new_attr3"
        mock_dispatch_data_info=Mock()
        dispatch_multiprocess(mock_run_param,mock_dispatch_data_info)
        mock_error.assert_called_once_with(f'can not find suitable call api:{mock_run_param.aten_api}')
        
    @patch('msprobe.pytorch.online_dispatch.dump_compare.dispatch_workflow')
    def test_dispatch_multiprocess_should_workflow_when_right_api_input(self,mock_workflow):
        mock_run_param = Mock()
        mock_run_param.func_namespace="aten"
        mock_run_param.aten_api="add"
        mock_run_param.aten_api_overload_name="Scalar"
        mock_dispatch_data_info=Mock()
        mock_workflow.return_value=1
        dispatch_multiprocess(mock_run_param,mock_dispatch_data_info)
        mock_workflow.assert_called_once_with(mock_run_param,mock_dispatch_data_info)

    @patch('msprobe.core.common.log.BaseLogger.error')
    def test_error_call(self,mock_error):
        error_call("messages")
        mock_error.assert_called_once_with("multiprocess messages")

    def test_save_csv(self):
        save_csv([[self.data]],[2],self.csv_path)
        df = pd.read_csv(self.csv_path)
        df_gt = pd.DataFrame.from_dict(self.data_gt, orient='index').T
        self.assertTrue((df.all()==df_gt.all()).all())