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
from ..utils.global_state import ADD_MATCH_KEYS


class MatchNodesController:

    @staticmethod
    def process_md5_task_add(graph_data, npu_node_name, bench_node_name):
        npu_match_nodes_list = graph_data.get('npu_match_nodes', {})
        bench_match_nodes_list = graph_data.get('bench_match_nodes', {})
        npu_node_data = graph_data.get('NPU', {}).get('node', {}).get(npu_node_name, {})
        bench_node_data = graph_data.get('Bench', {}).get('node', {}).get(bench_node_name, {})
        # 去除节点名称前缀
        npu_input_data = GraphUtils.remove_prefix(npu_node_data.get('input_data', {}), npu_node_name + '.')
        bench_input_data = GraphUtils.remove_prefix(bench_node_data.get('input_data', {}), bench_node_name + '.')
        npu_output_data = GraphUtils.remove_prefix(npu_node_data.get('output_data', {}), npu_node_name + '.')
        bench_output_data = GraphUtils.remove_prefix(bench_node_data.get('output_data', {}), bench_node_name + '.')
        # 计算精度误差
        precision_input_error = MatchNodesController.calculate_md5_diff(npu_input_data, bench_input_data)
        precision_output_error = MatchNodesController.calculate_md5_diff(npu_output_data, bench_output_data)
        precision_error = precision_input_error and precision_output_error
        # 在原始数据上，添加匹配节点，和匹配节点信息
        npu_node_data['matched_node_link'] = [bench_node_name]
        bench_node_data['matched_node_link'] = [npu_node_name]
        npu_node_data['data']['precision_index'] = precision_error
        # 后端维护一个匹配节点列表，前端展示
        npu_match_nodes_list[npu_node_name] = bench_node_name
        bench_match_nodes_list[bench_node_name] = npu_node_name
        graph_data['npu_match_nodes'] = npu_match_nodes_list
        graph_data['bench_match_nodes'] = bench_match_nodes_list
        return {
            'success': True,
            'data': {
                'precision_error': precision_error,
            },
        }

    @staticmethod
    def process_md5_task_delete(graph_data, npu_node_name, bench_node_name):
        npu_match_nodes_list = graph_data.get('npu_match_nodes', {})
        bench_match_nodes_list = graph_data.get('bench_match_nodes', {})
        npu_node_data = graph_data.get('NPU', {}).get('node', {}).get(npu_node_name, {})
        bench_node_data = graph_data.get('Bench', {}).get('node', {}).get(bench_node_name, {})
        # 在原始数据上，删除匹配节点，和匹配节点信息
        npu_node_data['matched_node_link'] = []
        bench_node_data['matched_node_link'] = []
        # 后端维护一个匹配节点列表，前端展示
        try:
            del npu_node_data['data']['precision_index']
            del npu_match_nodes_list[npu_node_name]
            del bench_match_nodes_list[bench_node_name]
        except KeyError:
            return {
                'success': False,
                'error': "操作失败：删除节点信息失败",
            }
        graph_data['npu_match_nodes'] = npu_match_nodes_list
        graph_data['bench_match_nodes'] = bench_match_nodes_list
        return {
            'success': True,
            'data': {},
        }

    @staticmethod
    def process_summary_task_add(graph_data, npu_node_name, bench_node_name):
        npu_match_nodes_list = graph_data.get('npu_match_nodes', {})
        bench_match_nodes_list = graph_data.get('bench_match_nodes', {})
        npu_node_data = graph_data.get('NPU', {}).get('node', {}).get(npu_node_name, {})
        bench_node_data = graph_data.get('Bench', {}).get('node', {}).get(bench_node_name, {})
        # 计算统计误差
        intput_statistical_diff = MatchNodesController.calculate_statistical_diff(
            npu_node_data.get('input_data'), bench_node_data.get('input_data'), npu_node_name, bench_node_name
        )
        output_statistical_diff = MatchNodesController.calculate_statistical_diff(
            npu_node_data.get('output_data'), bench_node_data.get('output_data'), npu_node_name, bench_node_name
        )
        # 计算精度误差
        precision_error = MatchNodesController.calculate_max_relative_error(output_statistical_diff)
        # 有一个没有匹配上，则认为匹配失败
        if not intput_statistical_diff or not output_statistical_diff:
            return {
                'success': False,
                'error': '输入或输出统计误差值为空(Input and output statistical error calculation failed)',
            }

        if precision_error == -1:
            return {
                'success': False,
                'error': '输出统计误差值为空，计算精度误差失败(Calculation of precision error failed)',
            }
        # 在原始数据上，添加匹配节点，和匹配节点信息
        npu_node_data['matched_node_link'] = [bench_node_name]
        bench_node_data['matched_node_link'] = [npu_node_name]
        MatchNodesController.update_graph_node_data(npu_node_data.get('input_data'), intput_statistical_diff)
        MatchNodesController.update_graph_node_data(npu_node_data.get('output_data'), output_statistical_diff)
        npu_node_data['data']['precision_index'] = precision_error
        # 后端维护一个匹配节点列表，前端展示
        npu_match_nodes_list[npu_node_name] = bench_node_name
        bench_match_nodes_list[bench_node_name] = npu_node_name
        graph_data['npu_match_nodes'] = npu_match_nodes_list
        graph_data['bench_match_nodes'] = bench_match_nodes_list
        return {
            'success': True,
            'data': {
                'precision_error': precision_error,
                'intput_statistical_diff': intput_statistical_diff,
                'output_statistical_diff': output_statistical_diff,
            },
        }

    @staticmethod
    def process_summary_task_delete(graph_data, npu_node_name, bench_node_name):
        npu_match_nodes_list = graph_data.get('npu_match_nodes', {})
        bench_match_nodes_list = graph_data.get('bench_match_nodes', {})
        npu_node_data = graph_data.get('NPU', {}).get('node', {}).get(npu_node_name, {})
        bench_node_data = graph_data.get('Bench', {}).get('node', {}).get(bench_node_name, {})
        # 在原始数据上，删除匹配节点，和匹配节点信息
        npu_node_data['matched_node_link'] = []
        bench_node_data['matched_node_link'] = []

        MatchNodesController.delete_matched_node_data(npu_node_data.get('input_data'))
        MatchNodesController.delete_matched_node_data(npu_node_data.get('output_data'))
        # 后端维护一个匹配节点列表，前端展示
        try:
            del npu_node_data['data']['precision_index']
            del npu_match_nodes_list[npu_node_name]
            del bench_match_nodes_list[bench_node_name]
        except KeyError:
            return {
                'success': False,
                'error': "操作失败：删除节点信息失败",
            }
        graph_data['npu_match_nodes'] = npu_match_nodes_list
        graph_data['bench_match_nodes'] = bench_match_nodes_list
        return {
            'success': True,
            'data': {},
        }

    @staticmethod
    def calculate_statistical_diff(npu_data, bench_data, npu_node_name, bench_node_name):
        result = {}
        # 去除节点名称前缀并转化为列表形式
        npu_data_simple = list(GraphUtils.remove_prefix(npu_data, npu_node_name + '.').values())
        bench_data_simple = list(GraphUtils.remove_prefix(bench_data, bench_node_name + '.').values())
        npu_data_keys = list(GraphUtils.remove_prefix(npu_data, npu_node_name + '.').keys())
        # 使用 zip 只对比最短的列表长度
        for npu_values, bench_values in zip(npu_data_simple, bench_data_simple):
            npu_max = GraphUtils.convert_to_float(npu_values.get('Max', float('nan')))
            bench_max = GraphUtils.convert_to_float(bench_values.get('Max', float('nan')))
            npu_min = GraphUtils.convert_to_float(npu_values.get('Min', float('nan')))
            bench_min = GraphUtils.convert_to_float(bench_values.get('Min', float('nan')))
            npu_norm = GraphUtils.convert_to_float(npu_values.get('Norm', float('nan')))
            bench_norm = GraphUtils.convert_to_float(bench_values.get('Norm', float('nan')))
            npu_mean = GraphUtils.convert_to_float(npu_values.get('Mean', float('nan')))
            bench_mean = GraphUtils.convert_to_float(bench_values.get('Mean', float('nan')))

            # Calculate absolute differences
            max_diff_abs = abs(npu_max - bench_max)
            min_diff_abs = abs(npu_min - bench_min)
            norm_diff_abs = abs(npu_norm - bench_norm)
            mean_diff_abs = abs(npu_mean - bench_mean)

            # Calculate relative differences (avoid division by zero)
            max_diff_rel = abs(max_diff_abs / (bench_max if bench_max != 0 else 1))
            min_diff_rel = abs(min_diff_abs / (bench_min if bench_min != 0 else 1))
            norm_diff_rel = abs(norm_diff_abs / (bench_norm if bench_norm != 0 else 1))
            mean_diff_rel = abs(mean_diff_abs / (bench_mean if bench_mean != 0 else 1))

            # 将结果记录到字典中
            result[npu_node_name + '.' + npu_data_keys[len(result)]] = dict(
                zip(
                    ADD_MATCH_KEYS,
                    [
                        max_diff_abs,
                        min_diff_abs,
                        mean_diff_abs,
                        norm_diff_abs,
                        max_diff_rel,
                        min_diff_rel,
                        mean_diff_rel,
                        norm_diff_rel,
                    ],
                )
            )

        return result

    # 计算最大相对误差
    @staticmethod
    def calculate_max_relative_error(result):
        max_rel_error = -1
        for _, diff_values in result.items():
            max_diff_rel = diff_values.get('MaxRelativeErr', float('nan'))
            min_diff_rel = diff_values.get('MinRelativeErr', float('nan'))
            norm_diff_rel = diff_values.get('NormRelativeErr', float('nan'))
            mean_diff_rel = diff_values.get('MeanRelativeErr', float('nan'))

            max_rel_error_for_key = max(max_diff_rel, min_diff_rel, norm_diff_rel, mean_diff_rel)
            max_rel_error = max(max_rel_error, max_rel_error_for_key)

        return min(max_rel_error, 1)

    @staticmethod
    def calculate_md5_diff(npu_data, bench_data):
        # 对比每个NPU和Bench所有数据md值，如果有一个不一样则返回0,否则返回1
        for npu_key, bench_key in zip(npu_data, npu_data):
            npu_md5 = npu_data[npu_key].get('md5', '')
            bench_md5 = bench_data[bench_key].get('md5', '')
            if npu_md5 != bench_md5:
                return 0
        return 1

    @staticmethod
    def update_graph_node_data(graph_npu_node_data, statistical_diff):
        if not statistical_diff or not graph_npu_node_data:
            return
        for key, diff_values in statistical_diff.items():
            # 格式化相对误差字段
            for field in ['MaxRelativeErr', 'MinRelativeErr', 'NormRelativeErr', 'MeanRelativeErr']:
                diff_values[field] = GraphUtils.format_relative_err(diff_values.get(field, float('nan')))

            # 转换 absErr 为 NaN 字符串
            for field in ['MaxAbsErr', 'MinAbsErr', 'MeanAbsErr', 'NormAbsErr']:
                diff_values[field] = GraphUtils.nan_to_str(diff_values.get(field, float('nan')))
            if key in graph_npu_node_data:
                graph_npu_node_data[key].update(diff_values)
            else:
                graph_npu_node_data[key] = diff_values

    @staticmethod
    def delete_matched_node_data(graph_npu_node_data):
        keys_to_remove = ADD_MATCH_KEYS
        # 遍历graph_npu_node_data中的每个主键和对应的子字典
        for key, fild_obj in graph_npu_node_data.items():
            # 使用字典解析创建新的子字典，排除不需要的键
            graph_npu_node_data[key] = {
                sub_key: value 
                for sub_key, value in fild_obj.items() 
                if sub_key not in keys_to_remove
            }
