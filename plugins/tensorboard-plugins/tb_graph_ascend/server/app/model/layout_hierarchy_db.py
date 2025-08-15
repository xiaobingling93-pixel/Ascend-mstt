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
from .hierarchy_db import Hierarchy
from ..utils.global_state import NPU, BENCH, SINGLE
import time


class LayoutHierarchyModel:
    npu_hierarchy = None
    bench_hierarchy = None
    single_hierarchy = None

    hierarchy = {
        NPU: npu_hierarchy,
        BENCH: bench_hierarchy,
        SINGLE: single_hierarchy
    }    

    @staticmethod
    def change_expand_state(node_name, graph_type, repo, micro_step, rank, step):
        if node_name == 'root':
            start = time.perf_counter()
            LayoutHierarchyModel.hierarchy[graph_type] = Hierarchy(graph_type, repo, micro_step, rank, step)
            end = time.perf_counter()
            print("root change_expand_state time: ", end - start)
            
        elif LayoutHierarchyModel.hierarchy.get(graph_type, None):
            start = time.perf_counter()
            LayoutHierarchyModel.hierarchy[graph_type].update_graph_data(node_name)
            end = time.perf_counter()
            print("node update_graph_data time: ", end - start)
            start = time.perf_counter()
            LayoutHierarchyModel.hierarchy[graph_type].update_graph_shape()
            end = time.perf_counter()
            print("node update_graph_shape time: ", end - start)
            start = time.perf_counter()
            LayoutHierarchyModel.hierarchy[graph_type].update_graph_position()
            end = time.perf_counter()
            print("node update_graph_position time: ", end - start)
        else:
            return {}
        return LayoutHierarchyModel.hierarchy[graph_type].get_hierarchy()

    @staticmethod
    def update_hierarchy_data(graph_type):
        if LayoutHierarchyModel.hierarchy.get(graph_type, None):
            return LayoutHierarchyModel.hierarchy[graph_type].update_hierarchy_data()
        else:
            return {}
        
    @staticmethod
    def update_current_hierarchy_data(data):
        npu_update_data = [node for node in data if node['graph_type'] == NPU]
        bench_update_data = [node for node in data if node['graph_type'] == BENCH]
        if LayoutHierarchyModel.hierarchy.get(NPU, None):
             LayoutHierarchyModel.hierarchy[NPU].update_current_hierarchy_data(npu_update_data)
        if LayoutHierarchyModel.hierarchy.get(BENCH, None):
             LayoutHierarchyModel.hierarchy[BENCH].update_current_hierarchy_data(bench_update_data)
     
