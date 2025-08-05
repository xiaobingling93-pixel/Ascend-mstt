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
from .base_graph_service import GraphServiceStrategy
from ..repositories.graph_repo import GraphRepo
from ..utils.global_state import GraphState
from ..utils.graph_utils import GraphUtils
from ..utils.global_state import  NPU, BENCH, SINGLE
from ..utils.match_type import ResultType
from ..model_db.layout_hierarchy_model import LayoutHierarchyModel
from ..model_db.match_nodes_model import MatchNodesController
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
            return {'success': True, 'data':self.config_info}
        except Exception as e:
            logger.error(f"load graph config info failed, {e}")
            return {'success': False, 'error': 'load graph config info failed, {e}'}

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
        pass
    
    def search_node_by_overflow(self, meta_data, values):
        pass

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
            result = {}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            task = self.config_info.get('task')
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_match_children:
                    result = MatchNodesController.process_task_add_child_layer(npu_node_name,
                                                                               bench_node_name, task, step, rank)
                    return result
                else:
                    npu_node = self.repo.query_node_info(npu_node_name, NPU, rank, step)
                    bench_node = self.repo.query_node_info(bench_node_name, BENCH, rank, step)
                    graph_data = GraphUtils.convert_to_graph_json(npu_node, bench_node)
                    match_result:list[ResultType] = MatchNodesController.process_task_add(graph_data, npu_node_name, bench_node_name, task)
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
                        result = {'success': False, 'error': '未找到匹配的节点(Matched node not found) '}
                    return result
            else:
                return {'success': False, 'error': '任务类型不支持(Task type not supported) '}
        except Exception as e:
            return {'success': False, '操作失败': str(e), 'data': None}

    def add_match_nodes_by_config(self, config_file, meta_data):
        pass

    def delete_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_unmatch_children):
        try:
            if not self.conn:
                return {'success': False, 'error': 'database connection not init'}
            result = {}
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            task = self.config_info.get('task')
            
            # 根据任务类型计算误差
            if task == 'md5' or task == 'summary':
                if is_unmatch_children:
                    result = MatchNodesController.process_task_delete_child_layer(npu_node_name,
                                                                                  bench_node_name, task, step, rank)
                    return result
                else:
                    npu_node = self.repo.query_node_info(npu_node_name, NPU, rank, step)
                    bench_node = self.repo.query_node_info(bench_node_name, BENCH, rank, step)
                    graph_data = GraphUtils.convert_to_graph_json(npu_node, bench_node)
                    match_result = MatchNodesController.process_task_delete(graph_data, npu_node_name, bench_node_name, task)
                    # 遍历match_result，找到的success=true的节点，合并所有的data字段(数组类型)，合并到update_db_data
                    update_data = [node for item in match_result if item.get('success') is True 
                                   for node in item.get('data', [])]
                    if len(update_data) > 0:
                        # DB：更新数据库节点信息
                        print("update_data", update_data)
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
            return {'success': False, '操作失败': str(e), 'data': None}

    def save_data(self, meta_data):
        pass

    def update_colors(self, colors):
        pass

    def save_matched_relations(self):
        pass
