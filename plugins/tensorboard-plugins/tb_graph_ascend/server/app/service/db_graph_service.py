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
from ..utils.global_state import NPU_PREFIX, BENCH_PREFIX, NPU, BENCH, SINGLE
from ..model_db.layout_hierarchy_model import LayoutHierarchyModel
from tensorboard.util import tb_logging
logger = tb_logging.get_logger()


class DbGraphService(GraphServiceStrategy):

    def __init__(self, run_path, tag):
        super().__init__(run_path, tag)
        runs = GraphState.get_global_value('runs')
        db_path = os.path.join(runs.get(self.run), f"{tag}.vis.db")
        self.repo = GraphRepo(db_path)

    def load_graph_data(self):
        pass

    def load_graph_config_info(self):
        try:
            config_info = self.repo.query_config_info()
            return {'success': True, 'data': config_info}
        except Exception as e:
            logger.error(f"load graph config info failed, {e}")
            return {'success': False, 'error': 'load graph config info failed, {e}'}

    def load_graph_all_node_list(self, meta_data):
        pass    

    def change_node_expand_state(self, node_info, meta_data):
        try:
            config_info = self.repo.query_config_info()
            graph_type = node_info.get('nodeType')
            node_name = node_info.get('nodeName')
            micro_step = meta_data.get('microStep')
            rank = meta_data.get('rank')
            step = meta_data.get('step')
            # 单图
            if config_info.get('isSingleGraph'):
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
    
    def get_node_info(self, node_info, meta_data):
        pass

    def add_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_match_children):
        pass

    def add_match_nodes_by_config(self, config_file, meta_data):
        pass

    def delete_match_nodes(self, npu_node_name, bench_node_name, meta_data, is_unmatch_children):
        pass

    def save_data(self, meta_data):
        pass

    def update_colors(self, colors):
        pass

    def save_matched_relations(self):
        pass
