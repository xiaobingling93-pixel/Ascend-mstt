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
from .base_graph_service import GraphServiceStrategy


class DbGraphService(GraphServiceStrategy):
    def __init__(self, run_path, tag):
        super().__init__(run_path, tag)

    def load_graph_data(self):
        pass

    def load_graph_config_info(self):
        pass

    def load_graph_all_node_list(self, meta_data):
        pass

    def change_node_expand_state(self, node_info, meta_data):
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
