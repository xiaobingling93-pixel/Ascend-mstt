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
from ..utils.global_state import ADD_MATCH_KEYS, MODULE
from ..utils.global_state import GraphState


class MatchNodesController:

    @staticmethod
    def is_same_node_type(graph_data, npu_node_name, bench_node_name):
        npu_node_type = graph_data.get('NPU', {}).get('node', {}).get(npu_node_name, {}).get('node_type')
        bench_node_type = graph_data.get('Bench', {}).get('node', {}).get(bench_node_name, {}).get('node_type')
        if npu_node_type is None or bench_node_type is None or npu_node_type != bench_node_type:
            return False
        return True

    @staticmethod
    def process_task_add(graph_data, npu_node_name, bench_node_name, task):
        result = {}
        if task == 'md5':
            result = MatchNodesController.process_md5_task_add(graph_data, npu_node_name, bench_node_name)
        elif task == 'summary':
            result = MatchNodesController.process_summary_task_add(graph_data, npu_node_name, bench_node_name)
        else:
            result = {
                'success': False,
                'error': 'task类型错误'
            }
        return result

    @staticmethod
    def process_task_delete(graph_data, npu_node_name, bench_node_name, task):
        result = {}
        if task == 'md5':
            result = MatchNodesController.process_md5_task_delete(graph_data, npu_node_name, bench_node_name)
        elif task == 'summary':
            result = MatchNodesController.process_summary_task_delete(graph_data, npu_node_name, bench_node_name)
        else:
            result = {
                'success': False,
                'error': 'task类型错误'
            }
        return result

    @staticmethod
    def process_task_add_child_layer_by_config(graph_data, match_node_links, task):
        # 根据配置文件中的匹配关系，批量调用 process_task_add
        result = {}
        match_reslut = []
        for npu_node_name, bench_node_name in match_node_links.items():
            res = MatchNodesController.process_task_add(graph_data, npu_node_name, bench_node_name, task)
            match_reslut.append(res.get('success'))

        config_data = GraphState.get_global_value("config_data")
        
        result['success'] = True
        result['data'] = {
            'matchReslut': match_reslut,
            'npuMatchNodes': config_data.get('npuMatchNodes', {}),
            'benchMatchNodes': config_data.get('benchMatchNodes', {}),
            'npuUnMatchNodes': config_data.get('npuUnMatchNodes', []),
            'benchUnMatchNodes': config_data.get('benchUnMatchNodes', [])
        }
        return result

    @staticmethod
    def process_task_add_child_layer(graph_data, npu_node_name, bench_node_name, task):
        if not all([graph_data, npu_node_name, bench_node_name, task]):
            return {'success': False, 'error': '参数错误'}
     
        if not MatchNodesController.is_same_node_type(graph_data, npu_node_name, bench_node_name):
            return {
                'success': False,
                'error': '节点类型不一致,无法添加匹配关系'
            }
        npu_nodes = graph_data.get('NPU', {}).get('node', {})
        bench_nodes = graph_data.get('Bench', {}).get('node', {})
        # 1. 选中的目标节点和标杆侧节点添加匹配关系
        result = MatchNodesController.process_task_add(graph_data, npu_node_name, bench_node_name, task)

        # 2. 目标节点的子节点和标杆侧的子节点添加匹配关系

        def process_child_layer(npu_subnodes, bench_subnodes):
            npu_match_names = {}
            bench_match_names = {}
            # 2.1 提取目标侧节点的未匹配子节点名称和标杆侧节点的未匹配子节点名称
            for npu_subnode_name in npu_subnodes:
                node_info = npu_nodes.get(npu_subnode_name, {})
                if node_info.get('matched_node_link'):
                    continue
                node_type = node_info.get('node_type') if node_info else None
                # 2.2 处理节点名称，提取匹配信息
                match_name = extract_module_name(npu_subnode_name) if node_type == MODULE else extract_api_name(
                    npu_subnode_name)
                matched_name = f'{match_name}.{node_type}'
                npu_match_names.setdefault(matched_name, []).append(npu_subnode_name)

            for bench_subnode_name in bench_subnodes:
                node_info = bench_nodes.get(bench_subnode_name, {})
                if node_info.get('matched_node_link'):
                    continue
                node_type = node_info.get('node_type') if node_info else None
                match_name = extract_module_name(bench_subnode_name) if node_type == MODULE else extract_api_name(
                    bench_subnode_name)
                matched_name = f'{match_name}.{node_type}'
                bench_match_names.setdefault(matched_name, []).append(bench_subnode_name)

            common_keys = npu_match_names.keys() & bench_match_names.keys()
            # 2.3 取名称交集，添加匹配关系
            for key in common_keys:
                npu_subnode_list = npu_match_names.get(key, [])
                bench_subnode_list = bench_match_names.get(key, [])
                
                # 多个节点可能有一个module name
                for npu_subnode_name, bench_subnode_name in zip(npu_subnode_list, bench_subnode_list):
                    result = MatchNodesController.process_task_add(graph_data, npu_subnode_name, bench_subnode_name,
                                                                   task)
                    npu_subnodes = npu_nodes.get(npu_subnode_name, {}).get('subnodes', [])
                    bench_subnodes = bench_nodes.get(bench_subnode_name, {}).get('subnodes', [])
                    # 2.4 如果有子节点，递归调用2.1-2.4
                    if result.get('success') and npu_subnodes and bench_subnodes:
                        process_child_layer(npu_subnodes, bench_subnodes)

        def extract_module_name(subnode_name):
            splited_subnode_name = subnode_name.split('.')
            if len(splited_subnode_name) < 4:
                return ""
            module_name = splited_subnode_name[-4] if not splited_subnode_name[-4].isdigit() else splited_subnode_name[
                -5]
            return module_name

        def extract_api_name(subnode_name):
            splited_subnode_name = subnode_name.split('.')
            if len(splited_subnode_name) < 2:
                return ""
            api_name = splited_subnode_name[-2] if not splited_subnode_name[-2].isdigit() else splited_subnode_name[-3]
            return api_name

        npu_subnodes = npu_nodes.get(npu_node_name, {}).get('subnodes', [])
        bench_subnodes = bench_nodes.get(bench_node_name, {}).get('subnodes', [])
        if result.get('success') and npu_subnodes and bench_subnodes:
            process_child_layer(npu_subnodes, bench_subnodes)
        if result.get('success'):
            config_data = GraphState.get_global_value("config_data")
            result['data'] = {
                'npuMatchNodes': config_data.get('npuMatchNodes', {}),
                'benchMatchNodes': config_data.get('benchMatchNodes', {}),
                'npuUnMatchNodes': config_data.get('npuUnMatchNodes', []),
                'benchUnMatchNodes': config_data.get('benchUnMatchNodes', [])
            }
        return result

    @staticmethod
    def process_task_delete_child_layer(graph_data, npu_node_name, bench_node_name, task):
        if not all([graph_data, npu_node_name, bench_node_name, task]):
            return {'success': False, 'error': '参数错误'}

        npu_nodes = graph_data.get('NPU', {}).get('node', {})
        bench_nodes = graph_data.get('Bench', {}).get('node', {})
        # 1. 选中的目标节点和标杆侧节点添加匹配关系
        result = MatchNodesController.process_task_delete(graph_data, npu_node_name, bench_node_name, task)

        # 2. 目标节点的子节点和标杆侧的子节点添加匹配关系
        def process_child_layer(npu_child_nodes):
            for npu_subnode_name in npu_child_nodes:
                npu_subnode_info = npu_nodes.get(npu_subnode_name, {})
                matched_node_link = npu_subnode_info.get('matched_node_link', [])
                if not matched_node_link:
                    continue
                bench_subnode_name = matched_node_link[-1]
                result = MatchNodesController.process_task_delete(graph_data, npu_subnode_name, bench_subnode_name,
                                                                  task)
                npu_subnodes = npu_nodes.get(npu_subnode_name, {}).get('subnodes', [])
                bench_subnodes = bench_nodes.get(bench_subnode_name, {}).get('subnodes', [])
                # 2.4 如果有子节点，递归调用2.1-2.4
                if result.get('success') and npu_subnodes and bench_subnodes:
                    process_child_layer(npu_subnodes)

        npu_subnodes = npu_nodes.get(npu_node_name, {}).get('subnodes', [])
        bench_subnodes = bench_nodes.get(bench_node_name, {}).get('subnodes', [])
        if result.get('success') and npu_subnodes and bench_subnodes:
            process_child_layer(npu_subnodes)
        if result.get('success'):
            config_data = GraphState.get_global_value("config_data")
            
            result['data'] = {
                'npuMatchNodes': config_data.get('npuMatchNodes', {}),
                'benchMatchNodes': config_data.get('benchMatchNodes', {}),
                'npuUnMatchNodes': config_data.get('npuUnMatchNodes', []),
                'benchUnMatchNodes': config_data.get('benchUnMatchNodes', [])
            }
        return result

    @staticmethod
    def process_md5_task_add(graph_data, npu_node_name, bench_node_name):
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

        npu_graph_data = graph_data.get('NPU', {})
        bench_graph_data = graph_data.get('Bench', {})
        npu_node_data['matched_node_link'] = GraphUtils.get_parent_node_list(bench_graph_data, bench_node_name)
        bench_node_data['matched_node_link'] = GraphUtils.get_parent_node_list(npu_graph_data, npu_node_name)
        npu_node_data.setdefault('data', {})['precision_index'] = precision_error
        
        MatchNodesController.add_config_match_nodes(npu_node_name, bench_node_name)
        return {'success': True}

    @staticmethod
    def process_summary_task_add(graph_data, npu_node_name, bench_node_name):
        # 节点信息提取
        npu_node_data = graph_data.get('NPU', {}).get('node', {}).get(npu_node_name)
        bench_node_data = graph_data.get('Bench', {}).get('node', {}).get(bench_node_name)
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
        npu_graph_data = graph_data.get('NPU', {})
        bench_graph_data = graph_data.get('Bench', {})
        npu_node_data['matched_node_link'] = GraphUtils.get_parent_node_list(bench_graph_data, bench_node_name)
        bench_node_data['matched_node_link'] = GraphUtils.get_parent_node_list(npu_graph_data, npu_node_name)
        npu_node_data.setdefault('data', {})['precision_index'] = precision_error
        MatchNodesController.update_graph_node_data(npu_node_data.get('input_data'), intput_statistical_diff)
        MatchNodesController.update_graph_node_data(npu_node_data.get('output_data'), output_statistical_diff)
        MatchNodesController.add_config_match_nodes(npu_node_name, bench_node_name)
        return {'success': True}

    @staticmethod
    def process_md5_task_delete(graph_data, npu_node_name, bench_node_name):
        config_data = GraphState.get_global_value("config_data")
        npu_match_nodes_list = config_data.get('npuMatchNodes', {})
        bench_match_nodes_list = config_data.get('benchMatchNodes', {})
        if npu_match_nodes_list.get(npu_node_name) != bench_node_name or bench_match_nodes_list.get(
                bench_node_name) != npu_node_name:
            return {
                'success': False,
                'error': "操作失败：节点未匹配，请先匹配节点",
            }
        npu_node_data = graph_data.get('NPU', {}).get('node', {}).get(npu_node_name, {})
        bench_node_data = graph_data.get('Bench', {}).get('node', {}).get(bench_node_name, {})
        # 在原始数据上，删除匹配节点，和匹配节点信息
        npu_node_data['matched_node_link'] = []
        bench_node_data['matched_node_link'] = []
        # 后端维护一个匹配节点列表，前端展示
        del npu_node_data['data']['precision_index']
        MatchNodesController.delete_config_match_nodes(npu_node_name, bench_node_name)
        return {
            'success': True,
            'data': {},
        }

    @staticmethod
    def process_summary_task_delete(graph_data, npu_node_name, bench_node_name):
        config_data = GraphState.get_global_value("config_data")
        npu_match_nodes_list = config_data.get('npuMatchNodes', {})
        bench_match_nodes_list = config_data.get('benchMatchNodes', {})
        if npu_match_nodes_list.get(npu_node_name) != bench_node_name or bench_match_nodes_list.get(
                bench_node_name) != npu_node_name:
            return {
                'success': False,
                'error': "操作失败：节点未匹配，请先匹配节点",
            }
        npu_node_data = graph_data.get('NPU', {}).get('node', {}).get(npu_node_name, {})
        bench_node_data = graph_data.get('Bench', {}).get('node', {}).get(bench_node_name, {})
        # 在原始数据上，删除匹配节点，和匹配节点信息
        npu_node_data['matched_node_link'] = []
        bench_node_data['matched_node_link'] = []
        MatchNodesController.delete_matched_node_data(npu_node_data.get('input_data'))
        MatchNodesController.delete_matched_node_data(npu_node_data.get('output_data'))
        # 防止 KeyError 或 TypeError
        npu_node_data.get('data', {}).pop('precision_index', None)
        MatchNodesController.delete_config_match_nodes(npu_node_name, bench_node_name)
        return {
            'success': True,
            'data': {},
        }

    @staticmethod
    def add_config_match_nodes(npu_node_name, bench_node_name):
        config_data = GraphState.get_global_value("config_data")
        # 匹配列表和未匹配列表
        manual_match_nodes = config_data.setdefault('manualMatchNodes', {})
        npu_match_nodes_list = config_data.setdefault('npuMatchNodes', {})
        bench_match_nodes_list = config_data.setdefault('benchMatchNodes', {})
        npu_unmatehed_name_list = config_data.setdefault('npuUnMatchNodes', [])
        bench_unmatehed_name_list = config_data.setdefault('benchUnMatchNodes', [])
        # 更新匹配列表和未匹配列表
        if str(npu_node_name) in npu_unmatehed_name_list:
            npu_unmatehed_name_list.remove(str(npu_node_name))
        if str(bench_node_name) in bench_unmatehed_name_list:
            bench_unmatehed_name_list.remove(str(bench_node_name))
        manual_match_nodes[str(npu_node_name)] = str(bench_node_name)
        npu_match_nodes_list[str(npu_node_name)] = str(bench_node_name)
        bench_match_nodes_list[str(bench_node_name)] = str(npu_node_name)
        GraphState.set_global_value("config_data", config_data)

    @staticmethod
    def delete_config_match_nodes(npu_node_name, bench_node_name):
        config_data = GraphState.get_global_value("config_data")
        # 匹配列表和未匹配列表
        manual_match_nodes = config_data.setdefault('manualMatchNodes', {})
        npu_match_nodes_list = config_data.setdefault('npuMatchNodes', {})
        bench_match_nodes_list = config_data.setdefault('benchMatchNodes', {})
        npu_unmatehed_name_list = config_data.setdefault('npuUnMatchNodes', [])
        bench_unmatehed_name_list = config_data.setdefault('benchUnMatchNodes', [])
        # 更新匹配列表和未匹配列表
        if str(npu_node_name) in manual_match_nodes:
            del manual_match_nodes[str(npu_node_name)]
        if str(npu_node_name) in npu_match_nodes_list:
            del npu_match_nodes_list[str(npu_node_name)]
        if str(bench_node_name) in bench_match_nodes_list:
            del bench_match_nodes_list[str(bench_node_name)]
        npu_unmatehed_name_list.append(str(npu_node_name))
        bench_unmatehed_name_list.append(str(bench_node_name))
        GraphState.set_global_value("config_data", config_data)

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
        if npu_data == {} or bench_data == {}:
            return 0
        # 对比每个NPU和Bench所有数据md值，如果有一个不一样则返回0,否则返回1
        for npu_key, bench_key in zip(npu_data, bench_data):
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
            if not fild_obj or not isinstance(fild_obj, dict):
                continue
            # 使用字典解析创建新的子字典，排除不需要的键
            graph_npu_node_data[key] = {
                sub_key: value
                for sub_key, value in fild_obj.items()
                if sub_key not in keys_to_remove
            }
