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
import os
from .graph_service_base import GraphServiceStrategy
from ..repositories.graph_repo import GraphRepo
from ..utils.global_state import GraphState
from ..utils.graph_utils import GraphUtils
from ..utils.global_state import  NPU, BENCH, SINGLE
from ..model.layout_hierarchy_db import LayoutHierarchyModel
from ..model.match_nodes_model import MatchNodesController
from ..utils.global_state import MAX_RELATIVE_ERR, MIN_RELATIVE_ERR, MEAN_RELATIVE_ERR, NORM_RELATIVE_ERR
from tensorboard.util import tb_logging
logger = tb_logging.get_logger()


class DbGraphService(GraphServiceStrategy):

    def __init__(self, run_path, tag):
        super().__init__(run_path, tag)
        runs = GraphState.get_global_value('runs')
        db_path = os.path.join(runs.get(self.run), f"{tag}.vis.db")
        self.repo = GraphRepo(db_path)
        self.conn = self.repo.get_db_connection()
        self.config_info = {}

    def load_graph_data(self):
        if not self.repo:
            return {'success': False, 'error': 'database not init'}
        if not self.conn:
            self.conn = self.repo.get_db_connection()
        return {'success': True}

    def load_graph_config_info(self):
        try:
            self.config_info = self.repo.query_config_info()
            # 读取目录下配置文件列表
            modify_matched_files = GraphUtils.find_config_files(self.run)
            self.config_info['matchedConfigFiles'] = modify_matched_files or []
            return {'success': True, 'data':self.config_info}
        except Exception as e:
            logger.error(f"load graph config info failed, {e}")
            return {'success': False, 'error': f'load graph config info failed, {e}'}

    def load_graph_all_node_list(self, meta_data):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            micro_step = meta_data.get('microStep')
            result = {}
            if not self.config_info:
                self.config_info = self.repo.query_config_info() 
            # 单图
            if self.config_info.get('isSingleGraph'):
                # DB：根据graphType rank step micro_step 查询节点列表
                node_name_list = self.repo.query_node_name_list(rank, step, micro_step)
                result['npuNodeList'] = node_name_list
                return {'success': True, 'data': result} 
            # 双图
            else:
                config_data = GraphState.get_global_value("config_data")
                all_node_info = self.repo.query_all_node_info_in_one(rank, step, micro_step)
                config_data['npuMatchNodes'] = all_node_info.get('npu_match_node')
                config_data['benchMatchNodes'] = all_node_info.get('bench_match_node')
                config_data['npuUnMatchNodes'] = all_node_info.get('npu_unmatch_node')
                config_data['benchUnMatchNodes'] = all_node_info.get('bench_unmatch_node')
        
                result['npuMatchNodes'] = all_node_info.get('npu_match_node')
                result['benchMatchNodes'] = all_node_info.get('bench_match_node')
                result['npuUnMatchNodes'] = all_node_info.get('npu_unmatch_node')
                result['benchUnMatchNodes'] = all_node_info.get('bench_unmatch_node')
                result['npuNodeList'] = all_node_info.get('npu_node_list')
                result['benchNodeList'] = all_node_info.get('bench_node_list')
                
                return {'success': True, 'data': result}
        except Exception as e:
            logger.error(f"load graph all node list failed, {e}")
            return {'success': False, 'error': f'load graph all node list failed, {e}'}

    def change_node_expand_state(self, node_info, meta_data):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            graph_type = node_info.get('nodeType')
            node_name = node_info.get('nodeName')
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            micro_step = meta_data.get('microStep')
            # 单图
            if self.config_info.get('isSingleGraph'):
               hierarchy = LayoutHierarchyModel.change_expand_state(node_name, SINGLE, self.repo, micro_step, rank, step)
            # NPU
            elif graph_type == NPU:
               hierarchy = LayoutHierarchyModel.change_expand_state(node_name, NPU, self.repo, micro_step, rank, step)
            # 标杆
            elif graph_type == BENCH:
               hierarchy = LayoutHierarchyModel.change_expand_state(node_name, BENCH, self.repo, micro_step, rank, step)
            else:
                return {'success': True, 'data': {}}
            return {'success': True, 'data': hierarchy}
        except Exception as e:
            logger.error('节点展开或收起发生错误:' + str(e))
            return {'success': False, 'error': f'节点展开或收起发生错误', 'data': None}

    def search_node_by_precision(self, meta_data, values):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            micro_step = meta_data.get('microStep')
            
            is_filter_unmatch_nodes = True if '无匹配节点' in values else False
            if is_filter_unmatch_nodes:
                values.remove('无匹配节点')
            
            node_name_list = self.repo.query_node_list_by_precision(step, rank, micro_step, values, is_filter_unmatch_nodes)
            return {'success': True, 'data':node_name_list}
        except Exception as e:
            logger.error('节点搜索发生错误:' + str(e))
            return {'success': False, 'error': f'节点搜索发生错误', 'data': None}
    
    def search_node_by_overflow(self, meta_data, values):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            micro_step = meta_data.get('microStep')
            
            node_name_list = self.repo.query_node_list_by_overflow(step, rank, micro_step, values)
            return {'success': True, 'data':node_name_list}
        except Exception as e:
            logger.error('节点搜索发生错误:' + str(e))
            return {'success': False, 'error': f'节点搜索发生错误', 'data': None}

    def update_hierarchy_data(self, graph_type):
        if (graph_type == NPU or graph_type == BENCH):
            hierarchy = LayoutHierarchyModel.update_hierarchy_data(graph_type)
            return {'success': True, 'data': hierarchy}
        else:
            return {'success': False, 'error': '节点类型错误'}

    def get_node_info(self, node_info, meta_data):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            result = {}
            graph_type = node_info.get('nodeType')
            node_name = node_info.get('nodeName')
            rank = meta_data.get('rank')
            step = meta_data.get('step')
       
            if self.config_info.get('isSingleGraph') or graph_type == SINGLE:
                result['npu'] = self.repo.query_node_info(node_name, graph_type, rank, step)
            else:
                matched_node_type = BENCH if graph_type == NPU else NPU
                node = self.repo.query_node_info(node_name, graph_type, rank, step)
                matched_node_link = node.get('matched_node_link', []) 
                if isinstance(node.get('matched_node_link', []), list) and len(matched_node_link) > 0:
                    matched_node = self.repo.query_node_info(matched_node_link[-1], matched_node_type, rank, step)
                else:
                    matched_node = None
                result['npu'] = node if graph_type == NPU else matched_node
                result['bench'] = node if graph_type == BENCH else matched_node
            return {'success': True, 'data': result}
        except Exception as e:
            logger.error('获取节点信息失败:' + str(e))
            return {'success': False, 'error': '获取节点信息失败:' + str(e), 'data': None}

    def add_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_match_children):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            task = self.config_info.get('task')
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_match_children:
                    graph_data = self.repo.query_node_and_sub_nodes(npu_node_name, bench_node_name, rank, step)
                    match_result = MatchNodesController.process_task_add_child_layer(graph_data, npu_node_name,
                                                                               bench_node_name, task)
                else:
                    graph_data = self.repo.query_matched_nodes_info(npu_node_name, bench_node_name, rank, step)
                    match_result = MatchNodesController.process_task_add(graph_data, npu_node_name, bench_node_name, task)
                # 处理匹配结果
                update_data = [node for item in match_result if item.get('success') is True 
                                for node in item.get('data', [])]
                if len(update_data) > 0:
                    # DB：更新数据库节点信息
                    update_db_res = self.repo.update_nodes_info(update_data, rank, step)
                    if not update_db_res:
                        return {'success': False, 'error': '更新数据库失败(Update database failed) '}
                    # 视图：调用更新update_hirarchy方法，同步更新图
                    LayoutHierarchyModel.update_current_hierarchy_data(update_data)
                    # 返回：返回更新后的节点信息
                    config_data = GraphState.get_global_value("config_data")
                    result = {
                        'success': True,
                        'data': {
                            'npuMatchNodes': config_data.get('npuMatchNodes', {}),
                            'benchMatchNodes': config_data.get('benchMatchNodes', {}),
                            'npuUnMatchNodes': config_data.get('npuUnMatchNodes', []),
                            'benchUnMatchNodes': config_data.get('benchUnMatchNodes', [])
                        }
                    }     
                else:
                    result = {'success': False, 'error': '选择的节点不可匹配(Selected nodes do not match) '}
                return result
            else:
                return {'success': False, 'error': '任务类型不支持(Task type not supported)'}
        except Exception as e:
            logger.error(str(e))
            return {'success': False, 'error': str(e), 'data': None}

    def add_match_nodes_by_config(self, config_file_name, meta_data):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            result = {}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            task = self.config_info.get('task')
            match_node_links, error = GraphUtils.safe_load_data(meta_data.get('run'), config_file_name)
            graph_data = self.repo.query_matched_nodes_info_by_config(match_node_links, rank, step)
            if error:
                return {'success': False, 'error': '配置文件失败'}
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                match_result = MatchNodesController.process_task_add_child_layer_by_config(graph_data, match_node_links, task)
                update_data = [node for item in match_result if item.get('success') is True 
                                   for node in item.get('data', [])]
                if len(update_data) > 0:
                    update_db_res = self.repo.update_nodes_info(update_data, rank, step)
                    if not update_db_res:
                            return {'success': False, 'error': '更新数据库失败(Update database failed) '}
                    # 视图：调用更新update_hirarchy方法，同步更新图
                    LayoutHierarchyModel.update_current_hierarchy_data(update_data)
                    # 返回：返回更新后的节点信息
                    config_data = GraphState.get_global_value("config_data")
                    result['success'] = True
                    result['data'] = {
                        'matchReslut': match_result,
                        'npuMatchNodes': config_data.get('npuMatchNodes', {}),
                        'benchMatchNodes': config_data.get('benchMatchNodes', {}),
                        'npuUnMatchNodes': config_data.get('npuUnMatchNodes', []),
                        'benchUnMatchNodes': config_data.get('benchUnMatchNodes', [])
                    }
                else:
                    result = {'success': False, 'error': '选择的节点不可匹配(Selected nodes do not match) '}
                return result
            else:
                return {'success': False, 'error': '任务类型不支持(Task type not supported)'}
        except Exception as e:
            logger.error(str(e))
            return {'success': False, 'error': str(e), 'data': None}

    def delete_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_unmatch_children):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            task = self.config_info.get('task')
            
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_unmatch_children:
                    # DB：当前节点及其所有的子节点信息
                    graph_data = self.repo.query_node_and_sub_nodes(npu_node_name, bench_node_name, rank, step)
                    match_result = MatchNodesController.process_task_delete_child_layer(graph_data, npu_node_name,
                                                                                  bench_node_name, task)
                else:
                    graph_data = self.repo.query_matched_nodes_info(npu_node_name, bench_node_name, rank, step)
                    match_result = MatchNodesController.process_task_delete(graph_data, npu_node_name, bench_node_name, task)
                # 遍历match_result，找到的success=true的节点，合并所有的data字段(数组类型)，合并到update_db_data
                update_data = [node for item in match_result if item.get('success') is True 
                                for node in item.get('data', [])]
                if len(update_data) > 0:
                    # DB：更新数据库节点信息
                    update_db_res = self.repo.update_nodes_info(update_data, rank, step)
                    if not update_db_res:
                        return {'success': False, 'error': '更新数据库失败(Update database failed) '}
                    # 视图：调用更新update_hirarchy方法，同步更新图
                    LayoutHierarchyModel.update_current_hierarchy_data(update_data)
                    # 返回：返回更新后的节点信息
                    config_data = GraphState.get_global_value("config_data")
                    result = {
                        'success': True,
                        'data': {
                            'npuMatchNodes': config_data.get('npuMatchNodes', {}),
                            'benchMatchNodes': config_data.get('benchMatchNodes', {}),
                            'npuUnMatchNodes': config_data.get('npuUnMatchNodes', []),
                            'benchUnMatchNodes': config_data.get('benchUnMatchNodes', [])
                        }
                    }     
                else:
                    result = {'success': False, 'error': '未找到可匹配的节点(Matched node not found) '}
                return result
            else:
                return {'success': False, 'error': '任务类型不支持(Task type not supported) '}
        except Exception as e:
            logger.error('delete_match_nodes error: {}'.format(e))
            return {'success': False, '操作失败': str(e), 'data': None}
    
    def update_precision_error(self, meta_data, filter_value):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}

            rank = meta_data.get('rank')
            step = meta_data.get('step')
            npu_node_list = self.repo.query_node_info_by_data_source(step, rank, 'NPU')
            update_data = []
            for _, node_info in npu_node_list.items():
                output_statistical_diff = node_info.get('output_data', None)
                if not node_info.get('matched_node_link') or not output_statistical_diff:
                    continue
                max_rel_error = -1
                #  根据filter_value 的选择指标计算新的误差值
                for _, diff_values in output_statistical_diff.items():
                    filter_diff_rel = []
                    if MAX_RELATIVE_ERR in filter_value:
                        filter_diff_rel.append(diff_values.get('MaxRelativeErr'))
                    if MIN_RELATIVE_ERR in filter_value:
                        filter_diff_rel.append(diff_values.get('MinRelativeErr'))
                    if NORM_RELATIVE_ERR in filter_value:
                        filter_diff_rel.append(diff_values.get('NormRelativeErr'))
                    if MEAN_RELATIVE_ERR in filter_value:
                        filter_diff_rel.append(diff_values.get('MeanRelativeErr'))
                    # 过滤掉N/A
                    filter_diff_rel = [x for x in filter_diff_rel if x and x != 'N/A']
                    # 如果output指标中存在 Nan/inf/-inf, 直接标记为最大值
                    if "Nan" in filter_diff_rel or "inf" in filter_diff_rel or "-inf" in filter_diff_rel:
                        max_rel_error = 1
                        break
                    filter_diff_rel = [GraphUtils.convert_to_float(x) for x in filter_diff_rel]
                    max_rel_error_for_key = max(filter_diff_rel) if filter_diff_rel else 0
                    max_rel_error = max(max_rel_error, max_rel_error_for_key)
                if max_rel_error != -1:
                    update_data.append((
                        min(max_rel_error, 1),
                        step,
                        rank,
                        node_info.get('node_name')
                        
                    ))  
            if len(update_data) > 0:
                # DB：更新数据库节点信息
                update_db_res = self.repo.update_nodes_precision_error(update_data)
                if not update_db_res:
                    return {'success': False, 'error': '更新数据库失败(Update database failed) '}
                return {'success': True, 'data': {}}
            else:
                return {'success': False, 'error': '未找到可更新的节点(Matched node not found) '}
        except Exception as e:
            logger.error('update_precision_error error: {}'.format(e))
            return {'success': False, 'error': str(e), 'data': None}   

    def update_colors(self, colors):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            # DB：更新颜色
            update_db_res = self.repo.update_config_colors(colors)
            if not update_db_res:
                return {'success': False, 'error': '更新数据库失败(Update database failed) '}
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e), 'data': None}

    def save_matched_relations(self, meta_data):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            run = meta_data.get('run')
            tag = meta_data.get('tag')
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            # DB：根据step rank modify match_node_link查询已经修改的匹配成功的节点关系
            modify_matched_nodes_list = self.repo.query_modify_matched_nodes_list(rank, step)
            confilg_file_name = f"{tag}_{step}_{rank}.vis.config"
            _, error = GraphUtils.safe_save_data(modify_matched_nodes_list, run, confilg_file_name)
            if error:
                return {'success': False, 'error': error}
            else:
                return {'success': True, 'data': confilg_file_name}
       
        except Exception as e:
            logger.error('save_matched_relations error: {}'.format(e))
            return {'success': False, 'error': str(e), 'data': None}

