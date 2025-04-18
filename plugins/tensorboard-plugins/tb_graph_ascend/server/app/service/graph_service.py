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

from ..utils.graph_utils import GraphUtils
from ..utils.global_state import get_global_value
from ..controllers.match_nodes_controller import MatchNodesController


class GraphService:

    @staticmethod
    def get_node_info(node_info, meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': error_message}

        node_type = node_info.get('nodeType')
        node_name = node_info.get('nodeName')
        node_details = graph_data.get(node_type, {}).get('node', {}).get(node_name)
        return {'success': True, 'data': node_details}

    @staticmethod
    def add_match_nodes(npu_node_name, bench_node_name, meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': error_message}

        task = graph_data.get('task')
        # 根据任务类型计算误差
        if task == 'md5':
            result = MatchNodesController.process_md5_task_add(graph_data, npu_node_name, bench_node_name)
            return result
        elif task == 'summary':
            result = MatchNodesController.process_summary_task_add(graph_data, npu_node_name, bench_node_name)
            return result
        else:
            return {'success': False, 'error': '任务类型不支持(Task type not supported) '}

    @staticmethod
    def delete_match_nodes(npu_node_name, bench_node_name, meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': error_message}

        task = graph_data.get('task')
        # 根据任务类型计算误差
        if task == 'md5':
            result = MatchNodesController.process_md5_task_delete(graph_data, npu_node_name, bench_node_name)
            return result
        elif task == 'summary':
            result = MatchNodesController.process_summary_task_delete(graph_data, npu_node_name, bench_node_name)
            return result
        else:
            return {'success': False, 'error': '任务类型不支持(Task type not supported) '}

    @staticmethod
    def get_matched_state_list(meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': error_message}

        npu_match_nodes_list = graph_data.get('npu_match_nodes', {})
        bench_match_nodes_list = graph_data.get('bench_match_nodes', {})
        return {
            'success': True,
            'data': {
                'npu_match_nodes': npu_match_nodes_list,
                'bench_match_nodes': bench_match_nodes_list,
            },
        }

    @staticmethod
    def update_colors(run, colors):
        """Set new colors in jsondata."""
        try:
            first_run_tag = get_global_value("first_run_tag")
            first_file_data, error = GraphUtils.safe_load_data(run, first_run_tag)
            if error:
                return {'success': False, 'error': error}
            first_file_data['Colors'] = colors
            GraphUtils.safe_save_data(first_file_data, run, first_run_tag)
            return {'success': True, 'error': None, 'data': {}}
        except Exception as e:
            return {'success': False, 'error': str(e), 'data': None}

    @staticmethod
    def save_data(meta_data):
        graph_data, error_message = GraphUtils.get_graph_data(meta_data)
        if error_message:
            return {'success': False, 'error': error_message}

        run = meta_data.get('run')
        tag = meta_data.get('tag')
        try:
            GraphUtils.safe_save_data(graph_data, run, tag)
        except (ValueError, IOError, PermissionError) as e:
            return {'success': False, 'error': f"Error: {e}"}
        return {'success': True}
