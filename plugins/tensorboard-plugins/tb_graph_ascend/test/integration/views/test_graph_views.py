# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest
import json
from types import SimpleNamespace
from  pathlib import Path
from werkzeug.wrappers import Request
from werkzeug.test import EnvironBuilder
from data.test_case_factory import TestCaseFactory
from server.app.utils.global_state import GraphState
from server.app.views.graph_views import GraphView


@pytest.mark.integration
class TestGraphViews:
    
    captured = SimpleNamespace(status=None, headers=None)

    @staticmethod
    def start_response(status, response_headers):
        TestGraphViews.captured.status = status
        TestGraphViews.captured.headers = dict(response_headers)
        return lambda x: None  # 必须返回一个 writer callable

    @staticmethod
    def create_mock_request(path="/meta"):
        builder = EnvironBuilder(path=path)
        return builder.get_environ()

    @pytest.mark.parametrize("test_case",
                             [
                                {"case_id": "1",
                                 "description": "test_load_meta_dir",
                                 "excepted":{'st_test_cases': ['test_compare_resnet_data']}
                                }
                              ],
                             ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_load_meta_dir(self, test_case):
        logdir = Path(__file__).resolve().parent.parent.parent / 'data' / 'st_test_cases'
        GraphState.set_global_value('logdir', str(logdir))
        # 构造请求
        request = TestGraphViews.create_mock_request("/data/plugin/graph_ascend/load_meta_dir")
        response_iter = GraphView.load_meta_dir(request, TestGraphViews.start_response)
        excepted = test_case['excepted']
        # 获取响应内容
        response_body = json.loads(b''.join(response_iter).decode('utf-8'))
        assert response_body == excepted
        assert TestGraphViews.captured.status == "200 OK"
        assert TestGraphViews.captured.headers["Content-Type"] == "application/json"
        
    @pytest.mark.parametrize("test_case", [{"case_id": "2", "description": "test_load_graph_data"}], ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_load_graph_data(self, test_case):
        request = TestGraphViews.create_mock_request("/data/plugin/graph_ascend/load_graph_data?run=st_test_cases&tag=test_compare_resnet_data")
        response_iter = GraphView.load_graph_data(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        runs = GraphState.get_global_value('runs')
        current_run = GraphState.get_global_value('current_run')
        current_tag = GraphState.get_global_value('current_tag')
        assert current_run == runs.get('st_test_cases')
        assert current_tag == 'test_compare_resnet_data'
        assert TestGraphViews.captured.status == "200 OK"
        assert TestGraphViews.captured.headers["Content-Type"] == "text/event-stream; charset=utf-8"

    @pytest.mark.parametrize("test_case", TestCaseFactory.get_load_graph_config_info_cases(), ids=lambda c: f"{c['case_id']}:{c['description']}")
    def test_load_graph_config_info(self, test_case):
        request = TestGraphViews.create_mock_request("/data/plugin/graph_ascend/load_graph_config_info?run=st_test_cases&tag=test_compare_resnet_data")
        response_iter = GraphView.load_graph_config_info(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        excepted = test_case['expected']
        assert response_body == json.dumps(excepted)

    @pytest.mark.parametrize("test_case", TestCaseFactory.get_load_graph_all_node_list(), ids=lambda c: f"{c['case_id']}:{c['description']}")    
    def test_load_graph_all_node_list(self, test_case):
        request = TestGraphViews.create_mock_request("/data/plugin/graph_ascend/load_graph_all_node_list?run=st_test_cases&tag=test_compare_resnet_data")
        response_iter = GraphView.load_graph_all_node_list(request, TestGraphViews.start_response)
        response_body = b''.join(response_iter).decode('utf-8')
        excepted = test_case['expected']
        assert response_body == json.dumps(excepted)
    
