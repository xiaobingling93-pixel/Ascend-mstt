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

from msprobe.pytorch.visualization.graph.base_node import BaseNode
from msprobe.pytorch.visualization.graph.node_op import NodeOp
from msprobe.pytorch.visualization.utils import GraphConst
from msprobe.core.common.log import logger
from msprobe.core.common.const import Const


class Graph:
    def __init__(self, model_name):
        self.node_map = {}
        self.node_id_map = {}
        self.add_node(NodeOp.module, model_name)
        self.root = self.get_node(model_name)

    def __str__(self):
        infos = [f'{str(self.node_map.get(node_id))}' for node_id in self.node_map]
        info = "\n".join(infos)
        return info

    @staticmethod
    def match(graph_n, node_n, graph_b):
        """
        给定节点n，在另一个graph中匹配它对应的节点。前置条件是它的父节点匹配已经完成
        目前采用完全匹配的方式，后续可能在这里加入一定的模糊匹配逻辑
        返回匹配结果，匹配到的节点，以及祖先树。没匹配到则返回None, []
        """
        if not node_n or node_n.id not in graph_b.node_map:
            return None, []
        node_b = graph_b.node_map.get(node_n.id)
        if node_n != node_b:
            return None, []
        ancestors_n = node_n.get_ancestors()
        ancestors_b = node_b.get_ancestors()
        if ancestors_n != ancestors_b:
            return None, []
        return node_b, ancestors_n

    @staticmethod
    def mapping_match(node_n, graph_b, mapping_config):
        """
        根据映射配置对节点进行匹配
        """
        node_b = graph_b.node_map.get(mapping_config.get_mapping_string(node_n.id))
        if not node_b or not node_n.compare_mapping_node(node_b):
            return None, [], []
        ancestors_n = node_n.get_ancestors()
        ancestors_b = node_b.get_ancestors()
        return node_b, ancestors_n, ancestors_b

    @staticmethod
    def dfs(node, result):
        info = node.to_dict()
        result[node.id] = info
        for subnode in node.subnodes:
            Graph.dfs(subnode, result)

    @staticmethod
    def split_nodes_by_micro_step(nodes):
        """
        根据Module名称后缀数字, 区分一个step中的多个micro steps, 后缀数字相同代表节点属于同一个micro step.
        如果是非Module节点，分类到前一个Module节点所在的micro step.
        """
        result = {}
        default_id = 0
        result[default_id] = []

        for node in nodes:
            if node.op == NodeOp.module:
                micro_step_id = node.id.split(Const.SEP)[-1]
                try:
                    micro_step_id = int(micro_step_id)
                except ValueError:
                    logger.warning(f'The node id suffix {micro_step_id} is not a number, micro steps cannot be split.')
                    micro_step_id = 0
                if micro_step_id not in result:
                    default_id = micro_step_id
                    result[micro_step_id] = []
                result[micro_step_id].append(node)
            else:
                result[default_id].append(node)
        return result

    def add_node(self, node_op, node_id, up_node=None, id_accumulation=False):
        """
        在graph中进行节点的添加
        Args:
            node_op: 需要添加的节点类型
            node_id: 需要添加的节点id
            up_node：对应节点的父节点
            id_accumulation: 是否对传入的重复node_id进行累加
        """
        if node_id in self.node_map:
            if id_accumulation:
                self.node_id_map[node_id] = 0
            else:
                return node_id
        if id_accumulation:
            if node_id in self.node_id_map:
                self.node_id_map[node_id] += 1
            else:
                self.node_id_map[node_id] = 0
            node_id = f'{node_id}.{self.node_id_map[node_id]}'
        node = BaseNode(node_op, node_id, up_node)
        self.node_map[node_id] = node
        return node_id

    def get_node(self, node_id):
        """
        返回节点，不存在返回None
        """
        return self.node_map.get(node_id, None)

    def to_dict(self):
        """
        用于数据输出
        """
        result = {}
        result[GraphConst.JSON_ROOT_KEY] = self.root.id if self.root else 'None'
        result[GraphConst.JSON_NODE_KEY] = {}
        for node_id in self.node_map:
            info = self.node_map.get(node_id).to_dict()
            result[GraphConst.JSON_NODE_KEY][node_id] = info
        return result

    def paging_by_micro_step(self, graph_other=None):
        """
        给graph首层节点增加micro step标记，供前端分页展示，有助于在处理大规模图数据时进行优化和管理
        比对场景中，同步更新另一个图graph_other中相应节点的micro step信息
        Args:
            self: 当前graph
            graph_other: 可选参数，另一个graph
        Returns: 分批的数量
        """
        batches_n = Graph.split_nodes_by_micro_step(self.root.subnodes)
        for batch_number, nodes in batches_n.items():
            for node in nodes:
                node.micro_step_id = batch_number
                # 在graph_other中更新已匹配节点的micro_step_id
                if graph_other and node.matched_node_link:
                    node_other = graph_other.get_node(node.matched_node_link[-1])
                    if node_other:
                        node_other.micro_step_id = batch_number
        # 遍历graph_other根节点下的所有子节点，确保未匹配节点也有micro_step_id
        if graph_other:
            for node in graph_other.root.subnodes:
                if node.micro_step_id is None:
                    try:
                        micro_step_id = int(node.id.split(Const.SEP)[-1])
                    except ValueError:
                        micro_step_id = 0
                    node.micro_step_id = micro_step_id
        return len(batches_n)
