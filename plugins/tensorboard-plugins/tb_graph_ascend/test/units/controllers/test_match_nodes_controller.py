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
from server.app.controllers.match_nodes_controller import MatchNodesController
from server.app.utils.graph_utils import GraphUtils


class TestMatchNodesController:

    @staticmethod
    def test_match_node_controller(ut_test_case):
        op_type = ut_test_case.get("type")
        input_data = ut_test_case.get('input')
        expected = ut_test_case.get("expected")
         # 执行操作
        try:
            if op_type == "process_md5_task_add":
                result = MatchNodesController.process_md5_task_add(
                    graph_data=input_data.get("graph_data"),
                    npu_node_name=input_data.get("npu_node_name"),
                    bench_node_name=input_data.get("bench_node_name"),
                )
            elif op_type == "process_md5_task_delete":
                result = MatchNodesController.process_md5_task_delete(
                    graph_data=input_data.get("graph_data"),
                    npu_node_name=input_data.get("npu_node_name"),
                    bench_node_name=input_data.get("bench_node_name"),
                )
            elif op_type == "process_summary_task_add":
                
                result = MatchNodesController.process_summary_task_add(
                    graph_data=input_data.get("graph_data"),
                    npu_node_name=input_data.get("npu_node_name"),
                    bench_node_name=input_data.get("bench_node_name"),
                )
            elif op_type == "process_summary_task_delete":
                result = MatchNodesController.process_summary_task_delete(
                    npu_data=input_data.get("npu_data"),
                    bench_data=input_data.get("bench_data"),
                    npu_node_name=input_data.get("npu_node_name"),
                    bench_node_name=input_data.get("bench_node_name"),
                )
            elif op_type == "calculate_statistical_diff":
                result = MatchNodesController.calculate_statistical_diff(
                    npu_data=input_data.get("npu_data"),
                    bench_data=input_data.get("bench_data"),
                    npu_node_name=input_data.get("npu_node_name"),
                    bench_node_name=input_data.get("bench_node_name"),
                )
            elif op_type == "calculate_max_relative_error":
                result = MatchNodesController.calculate_max_relative_error(
                    result=input_data.get("result"),
                )
            elif op_type == "calculate_md5_diff":
                result = MatchNodesController.calculate_md5_diff(
                    npu_data=input_data.get("npu_data"),
                    bench_data=input_data.get("bench_data"),
                )
            elif op_type == "update_graph_node_data":
                result = MatchNodesController.update_graph_node_data(
                    graph_npu_node_data=input_data.get("graph_npu_node_data"),
                    statistical_diff=input_data.get("statistical_diff"),
                )
            elif op_type == "delete_matched_node_data":
                result = MatchNodesController.delete_matched_node_data(
                    graph_npu_node_data=input_data.get("graph_npu_node_data"),
                )
            else:
                return
        except Exception as e:
            result = {"error": type(e).__name__}
        
        # 验证结果
        assert result == expected, \
            f"Operation {op_type} failed on {ut_test_case}"
