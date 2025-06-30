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
import time
from .hierarchy import Hierarchy


class LayoutHierarchyController:
    npu_hierarchy = None
    bench_hierarchy = None
    single_hierarchy = None

    hierarchy = {
        'NPU': npu_hierarchy,
        'Bench': bench_hierarchy,
        'Single': single_hierarchy
    }

    @staticmethod
    def change_expand_state(node_name, graph_type, graph, micro_step):
        if node_name == 'root':
            LayoutHierarchyController.hierarchy[graph_type] = Hierarchy(graph_type, graph, micro_step)
        elif LayoutHierarchyController.hierarchy[graph_type]:
            LayoutHierarchyController.hierarchy[graph_type].update_graph_data(node_name, graph)
            LayoutHierarchyController.hierarchy[graph_type].update_graph_shape()
            LayoutHierarchyController.hierarchy[graph_type].update_graph_position()
        else:
            return {}
        return LayoutHierarchyController.hierarchy[graph_type].get_hierarchy()

    @staticmethod
    def update_hierarchy_data(graph_type):
        if LayoutHierarchyController.hierarchy.get(graph_type, None):
            return LayoutHierarchyController.hierarchy[graph_type].update_hierarchy_data()
        else:
            return {}
