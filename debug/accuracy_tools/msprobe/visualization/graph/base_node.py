# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.utils import Suggestions, GraphConst
from msprobe.visualization.builder.msprobe_adapter import format_node_data, compare_data, compare_mapping_data


class BaseNode:
    def __init__(self, node_op, node_id, up_node=None):
        self.op = node_op
        self.id = node_id
        self.data = {}
        self.output_data = {}
        self.input_data = {}
        self.upnode = None
        self.add_upnode(up_node)
        self.subnodes = []
        self.matched_node_link = []
        self.suggestions = {}
        self.stack_info = []
        self.micro_step_id = None

    def __str__(self):
        info = f'id:\t{self.id}'
        return info

    def __eq__(self, other):
        """
        用来判断两个节点是否可以被匹配上，认为结构上是否一致
        """
        if not compare_data(self.input_data, other.input_data):
            return False
        if not compare_data(self.output_data, other.output_data):
            return False
        return True

    def get_suggestions(self):
        """
        精度疑似有问题时，提供一些建议
        """
        if self.op == NodeOp.module:
            self.suggestions[GraphConst.SUGGEST_KEY] = Suggestions.Module
            self.suggestions[Suggestions.DUMP] = Suggestions.DUMP_URL
        elif self.op == NodeOp.function_api:
            self.suggestions[GraphConst.SUGGEST_KEY] = Suggestions.API
            self.suggestions[Suggestions.API_ACCURACY_CHECKER] = Suggestions.API_ACCURACY_CHECKER_URL

    def set_input_output(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def add_upnode(self, node):
        """
        绑定upnode，用于对两个节点进行上下级关联
        """
        if not node or node.id == self.id or self.upnode:
            return
        self.upnode = node
        node.subnodes.append(self)

    def add_link(self, node, ancestors):
        """
        在节点匹配成功后进行匹配数据的录入
        Args:
            node: 和self相互匹配的节点
            ancestors: 对面节点的祖先信息
        """
        self.matched_node_link = ancestors
        node.matched_node_link = ancestors

    def to_dict(self):
        """
        输出数据
        """
        result = {}
        result['id'] = self.id
        result['node_type'] = self.op.value
        result['data'] = self.data
        result['output_data'] = format_node_data(self.output_data)
        result['input_data'] = format_node_data(self.input_data)
        result['upnode'] = self.upnode.id if self.upnode else 'None'
        result['subnodes'] = [node.id for node in self.subnodes]
        result['matched_node_link'] = self.matched_node_link
        result['suggestions'] = self.suggestions
        result['stack_info'] = self.stack_info
        if self.micro_step_id is not None:
            result['micro_step_id'] = self.micro_step_id
        return result

    def get_ancestors(self):
        """
        获取节点所有祖先的列表
        """
        ancestors = []
        current_node = self.upnode
        while current_node:
            ancestors.append(current_node.id)
            current_node = current_node.upnode
        return list(reversed(ancestors))
