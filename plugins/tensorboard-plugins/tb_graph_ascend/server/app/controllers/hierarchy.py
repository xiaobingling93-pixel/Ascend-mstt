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
from tensorboard.util import tb_logging
from ..utils.global_state import NPU_PREFIX, BENCH_PREFIX, NPU, SINGLE, UNEXPAND_NODE, MODULE

logger = tb_logging.get_logger()

INNER_WIDTH = 50  # 内部节点宽度
INNER_HIGHT = 15  # 内部节点高度
HORIZONTAL_SPACING = 5  # 横向排列间距
VERTICAL_SPACING = 10  # 纵向排列间距
MAX_PER_ROW = 5  # 横向每行最大数


class Hierarchy:
    
    def __init__(self, graph_type, graph, micro_step):
        root_node_name = graph.get('root')
        node_info = graph.get('node', {}).get(root_node_name, {})
        name_prefix = NPU_PREFIX if graph_type == NPU else BENCH_PREFIX
        name_prefix = '' if graph_type == SINGLE else name_prefix
        self.name_prefix = name_prefix
        self.root_name = root_node_name
        self.graph_type = graph_type
        self.graph = graph
        self.micro_step_id = micro_step
        self.current_hierarchy = {
            root_node_name: self.get_basic_rende_info(root_node_name, node_info)
        }
        # 默认展开根节点
        self.update_graph_data(self.root_name, graph)
        self.update_graph_shape()
        self.update_graph_position()

    @staticmethod
    def measure_text_width(text):
        return len(text) * 6  # 假设每个字符宽度为6

    @staticmethod
    def extract_label_name(node_name, node_type):
        splited_subnode_name = node_name.split('.')
        splited_label = []
        # 在展开层级时，将父级层级名称相关去除，仅保留子节点本身名称信息
        # 如Module.layer1.1.relu.ReLU.forward.0中的父级名称Module.layer1.1去除，仅保留子级的relu.ReLU.forward.1
        # 如Module.layer4.0.BasicBlock.forward.0中的父级名称Module.1去除，仅保留子级的layer4.0.BasicBlock.forward.0
        if node_type == MODULE:
            if len(splited_subnode_name) < 4:
                return node_name
            splited_label = splited_subnode_name[-4:] if not splited_subnode_name[
                -4].isdigit() else splited_subnode_name[-5:]
        # 在展开层级时，将父级层级名称相关去除，仅保留API子节点本身名称信息，
        # 如 Module.layer1.1.ApiList.1 中的父级名称Module.layer1.1去除，仅保留子级的ApiList.1
        # 如 Module.layer1.1.ApiList.0.1 中的父级名称Module.layer1.1去除，仅保留子级的ApiList.0.1
        else:
            if len(splited_subnode_name) < 2:
                return node_name
            splited_label = splited_subnode_name[-2:] if not splited_subnode_name[
                -2].isdigit() else splited_subnode_name[-3:]
        return ('.').join(splited_label)

    def update_graph_data(self, node_name, graph):
        target_node = self.current_hierarchy.get(node_name, {})
        if node_name == self.root_name or target_node:
            self.process_click_expand(node_name, graph)
        else:  # 如果图中不存在该节点，说明是选中节点，需要递归展开该节点的所有父节点
            self.process_select_expand(node_name, graph)

    def process_click_expand(self, node_name, graph):
        target_node = self.current_hierarchy.get(node_name, {})
        target_node_children = target_node.get("children", [])
        if not target_node or not target_node_children:
            return
        if not target_node.get('expand', False):
            # 1.将target_node的expand置为true
            # 2.将node_name的子节点信息初始化，并添加到current_hierarchy中
            for subnode_name in target_node_children:
                if self.current_hierarchy.get(subnode_name):
                    continue
                node_info = graph.get('node', {}).get(subnode_name, {})
                render_info = self.get_basic_rende_info(subnode_name, node_info)
                self.current_hierarchy[subnode_name] = render_info
            target_node['expand'] = True
        else:
            target_node['expand'] = False if node_name != self.root_name else True  # 根节点默认展开

    def process_select_expand(self, node_name, graph):
        parent_node_name = graph.get('node', {}).get(node_name, {}).get("upnode")
        parent_node = self.current_hierarchy.get(parent_node_name)
        # 递归展开父节点
        while not parent_node or not parent_node.get('expand', False):
            if not parent_node:  # 如果父节点不存在，则初始化父节点
                node_info = graph.get('node', {}).get(parent_node_name)
                render_info = self.get_basic_rende_info(parent_node_name, node_info)
                self.current_hierarchy[parent_node_name] = render_info
            try:
                self.process_click_expand(parent_node_name, graph)  # 展开父节点
            except Exception as e:
                logger.error(f"Failed to expand parent node {parent_node_name}: {e}")
                break
            if parent_node_name == self.root_name:
                break
            parent_node_name = graph.get('node', {}).get(parent_node_name, {}).get("upnode")
            parent_node = self.current_hierarchy.get(parent_node_name)

    def update_graph_shape(self):
        self.resize_hierarchy(self.root_name)

    def update_graph_position(self):
        self.layout_hierarchy(self.root_name)

    def resize_hierarchy(self, current_name):
        node = self.current_hierarchy.get(current_name, {})
        if not node:
            return
        if node.get('nodeType') == UNEXPAND_NODE:
            # 不可展开叶子节点固定尺寸
            node['width'] = INNER_WIDTH
            node['height'] = INNER_HIGHT
            return
        if not node.get('expand', False):
            # 未展开的父节点按文字宽度
            node['width'] = Hierarchy.measure_text_width(node.get('label', '')) + HORIZONTAL_SPACING * 2  # 文字宽度 + 边距
            node['height'] = INNER_HIGHT
            return
        for child_name in node.get('children', []):
            self.resize_hierarchy(child_name)
        max_child_width = 0
        total_height = INNER_HIGHT
        for group_type, children in self.group_children(node.get('children')):
            if group_type == UNEXPAND_NODE:
                # 横向布局：分批处理每行最多MAX_PER_ROW个
                rows = [children[i:i + MAX_PER_ROW] for i in range(0, len(children), MAX_PER_ROW)]
                if (len(rows) > 1):
                    max_child_width = max(max_child_width,
                                          INNER_WIDTH * MAX_PER_ROW + (HORIZONTAL_SPACING * (MAX_PER_ROW - 1)))
                    total_height += ((INNER_HIGHT + VERTICAL_SPACING) * len(rows))
                else:
                    child_total_width = sum(self.current_hierarchy.get(child, {}).get('width', 0) for child in children)
                    spacing = HORIZONTAL_SPACING * (len(children) - 1)
                    max_child_width = max(max_child_width, child_total_width + spacing)
                    total_height += (INNER_HIGHT + VERTICAL_SPACING)

            else:
                # 纵向布局，计算子节点最大宽度
                # 更新 max_child_width：取所有子节点中最宽的一个
                child_max_width = max(self.current_hierarchy.get(child, {}).get('width', 0) for child in children)
                max_child_width = max(max_child_width, child_max_width)
                # 累加 total_height：所有子节点高度 + 垂直间距（每个子节点之间都有间距，所以是 len(children) 个间距）
                children_heights = sum(self.current_hierarchy.get(child, {}).get('height', 0) for child in children)
                vertical_spacing = VERTICAL_SPACING * len(children)
                total_height += (children_heights + vertical_spacing)

        # 最终尺寸计算
        node['width'] = max(
            max_child_width + HORIZONTAL_SPACING * 2,  # 子节点最大宽度 + 边距
            Hierarchy.measure_text_width(node.get('label', '')) + HORIZONTAL_SPACING * 2  # 保证文字可见
        )
        node['height'] = total_height + VERTICAL_SPACING  # 总高度 + 边距

    def layout_hierarchy(self, current_name):
        node = self.current_hierarchy.get(current_name, {})
        if not node or not node.get('children'):
            return
        parent_x = node.get('x')
        parent_width = node.get('width')
        current_y = node.get('y') + INNER_HIGHT + VERTICAL_SPACING  # 初始Y坐标

        # 按类型分组处理子节点
        for group_type, children in self.group_children(node.get('children')):
            if group_type == UNEXPAND_NODE:
                # 横向布局：分批处理每行最多MAX_PER_ROW个
                rows = [children[i:i + MAX_PER_ROW] for i in range(0, len(children), MAX_PER_ROW)]

                for row_children in rows:
                    # 计算该行总宽度
                    row_width = sum(
                        self.current_hierarchy.get(child_name, {}).get('width') for child_name in row_children)
                    row_width += HORIZONTAL_SPACING * (len(row_children) - 1)

                    # 居中对齐计算
                    start_x = parent_x + (parent_width - row_width) // 2
                    max_height = 0
                    current_x = start_x

                    # 布局该行所有子节点
                    for child_name in row_children:
                        child = self.current_hierarchy.get(child_name, {})
                        child['x'] = current_x
                        child['y'] = current_y
                        current_x += child.get('width') + HORIZONTAL_SPACING
                        max_height = max(max_height, child.get('height'))

                    # 更新Y坐标
                    current_y += max_height + VERTICAL_SPACING
            else:
                # 纵向布局：每个子节点单独占一行且居中
                for child_name in children:
                    child = self.current_hierarchy.get(child_name, {})
                    child['x'] = parent_x + (parent_width - child.get('width')) // 2
                    child['y'] = current_y
                    current_y += child.get('height') + VERTICAL_SPACING
                    if (child.get('expand', False)):
                        self.layout_hierarchy(child_name)  # 递归处理子节点

    def group_children(self, children):
        """将子节点按类型分组，连续同类型节点为一组"""
        groups = []
        current_group = []
        current_type = None

        for name in children:
            child = self.current_hierarchy.get(name, {})
            if not child:
                continue
            # 类型变化时创建新组
            if current_type is not None and child.get('nodeType') != current_type:
                groups.append((current_type, current_group))
                current_group = []
            current_type = child.get('nodeType')
            current_group.append(name)
        # 添加最后一组
        if current_group:
            groups.append((current_type, current_group))
        return groups

    def get_basic_rende_info(self, node_name, node_info):
        if not node_info:
            return {}
        label = node_name
        children = []
        if node_name == self.root_name:  # 根节点，根据micro_step获取子节点
            target_node_children = node_info.get('subnodes', [])
            for subnode_name in target_node_children:
                child_node = self.graph.get('node', {}).get(subnode_name, {})
                child_micro_step_id = child_node.get('micro_step_id', -1)  # 如果子节点不包含micro_step_id，则默认为-1，直接添加
                is_append_all_node = int(self.micro_step_id) == -1 or child_micro_step_id == -1
                is_append_split_node = int(self.micro_step_id) != -1 and int(child_micro_step_id) == int(
                    self.micro_step_id)
                if is_append_all_node or is_append_split_node:
                    children.append(subnode_name)
        else:
            children = node_info.get('subnodes', [])
        if node_info.get('upnode', '') != self.root_name:  # 首层节点不处理显示内容
            label = Hierarchy.extract_label_name(node_name, node_info.get('node_type'))
        render_info = {
            'x': 0,
            'y': 0,
            'width': Hierarchy.measure_text_width(node_name) + HORIZONTAL_SPACING * 2,
            'height': INNER_HIGHT,
            'expand': False,
            'isRoot': node_name == self.root_name,
            'parentNode': node_info.get('upnode', ''),
            'label': label,
            'name': self.name_prefix + node_name,
            'children': children,
            'nodeType': node_info.get('node_type') if node_info.get("subnodes") else UNEXPAND_NODE,
            'matchedNodeLink': node_info.get('matched_node_link', []),
            'precisionIndex': node_info.get('data', {}).get('precision_index', "NaN"),  # 精度
            'overflowLevel': node_info.get('data', {}).get('overflow_level', "NaN"),  # 溢出
            'matchedDistributed': node_info.get('matched_distributed', {}),
        }

        return render_info

    # 获取连通图
    def get_connected_graph(self, name, result, new_hierarchy):
        node = self.current_hierarchy.get(name, {})
        if (not node): 
            return
        filtered_value = {k: v for k, v in node.items() if k != "children"}
        result[name] = filtered_value
        new_hierarchy[name] = node
        # 递归处理子节点
        if (node.get('expand')):
            for child_name in node.get('children', []):
                self.get_connected_graph(child_name, result, new_hierarchy)

    def update_hierarchy_data(self):
        for node_name, node_info in self.current_hierarchy.items():
            graph_node_info = self.graph.get('node', {}).get(node_name, {})
            node_info['matchedNodeLink'] = graph_node_info.get('matched_node_link', [])
            node_info['precisionIndex'] = graph_node_info.get('data', {}).get('precision_index', "NaN"),  # 精度
        return self.current_hierarchy

    def get_hierarchy(self):
        result = {}
        new_hierarchy = {}
        # 遍历当前层级结构
        self.get_connected_graph(self.root_name, result, new_hierarchy)
        self.current_hierarchy = new_hierarchy  # 折叠之后的需要重新展开
        return result
