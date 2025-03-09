# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import re

from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import load_json
from msprobe.visualization.builder.msprobe_adapter import get_input_output
from msprobe.visualization.builder.msprobe_adapter import op_patterns
from msprobe.visualization.graph.graph import Graph
from msprobe.visualization.graph.node_op import NodeOp
from msprobe.visualization.utils import save_json_file, GraphConst


class GraphBuilder:
    backward_pattern = re.compile(r"(\.backward\.)(\d+)$")
    forward_pattern = re.compile(r"(\.forward\.)(\d+)$")
    # 匹配以大写字母开头，后接任意字母，并以Template(结尾
    template_pattern = re.compile(r'\b[A-Z][a-zA-Z]*Template\(')

    @staticmethod
    def build(construct_path, data_path, stack_path, model_name='DefaultModel', complete_stack=False):
        """
        GraphBuilder的对外提供的构图方法
        Args:
            construct_path: construct.json路径
            data_path: dump.json路径
            stack_path: stack.json路径
            model_name: 模型名字，依赖外部输入
            complete_stack: 完整的堆栈信息
        Returns: Graph，代表图的数据结构
        """
        construct_dict = load_json(construct_path)
        dump_dict = load_json(data_path)
        stack_dict = load_json(stack_path)
        if not complete_stack:
            GraphBuilder._simplify_stack(stack_dict)
        data_dict = dump_dict.get(GraphConst.DATA_KEY, {})
        graph = Graph(model_name, data_path=dump_dict.get('dump_data_dir', ''), dump_data=data_dict)
        GraphBuilder._init_nodes(graph, construct_dict, data_dict, stack_dict)
        GraphBuilder._collect_apis_between_modules(graph)
        return graph

    @staticmethod
    def to_json(filename, config):
        """
        将graph导出成.vis文件的接口
        """
        result = {}
        if config.graph_b:
            result[GraphConst.JSON_NPU_KEY] = config.graph_n.to_dict()
            result[GraphConst.JSON_BENCH_KEY] = config.graph_b.to_dict()
        else:
            result = config.graph_n.to_dict()
        if config.tool_tip:
            result[GraphConst.JSON_TIP_KEY] = config.tool_tip
        if config.node_colors:
            result[GraphConst.COLORS] = config.node_colors
        if config.micro_steps:
            result[GraphConst.MICRO_STEPS] = config.micro_steps
        if config.task:
            result[GraphConst.JSON_TASK_KEY] = config.task
        result[GraphConst.OVERFLOW_CHECK] = config.overflow_check
        save_json_file(filename, result)

    @staticmethod
    def _simplify_stack(stack_dict):
        """
        精简堆栈内容，模块级保留包含"模块名("的堆栈，api级保留"xxxTemplate("的下一行堆栈

        例如模块 Module.layer3.0.bn2.BatchNorm2d.forward.0，模块名为bn2，匹配"bn2("，
        保留堆栈"File /home/models/resnet.py, line 97, in forward, \n out = self.bn2(out)"

        例如Api Tensor.__iadd__.4.forward，堆栈为：
        "File /home/wrap_tensor.py, line 61,  return TensorOPTemplate(op_name, hook)(*args, **kwargs)",
        "File /home/torchvision/models/resnet.py, line 102, in forward, \n out += identity",
        匹配到第一行的"TensorOPTemplate("，保留下一行堆栈
        """
        module_pattern = re.compile(op_patterns[0])
        for dump_name, stack_list in stack_dict.items():
            if not isinstance(stack_list, list):
                continue
            if module_pattern.match(dump_name):
                parts = dump_name.split(Const.SEP)
                if len(parts) < abs(Const.LAYER_NAME_INDEX):
                    continue
                module_name = parts[Const.LAYER_NAME_INDEX]
                for stack in stack_list:
                    if re.search(module_name + r'\(', stack):
                        stack_list = [stack]
                        break
            else:
                for index, stack in enumerate(stack_list):
                    if GraphBuilder.template_pattern.search(stack) and index < len(stack_list) - 1:
                        stack_list = [stack_list[index + 1]]
                        break
            stack_dict[dump_name] = stack_list

    @staticmethod
    def _handle_backward_upnode_missing(construct_dict, subnode_id, upnode_id):
        """
        如果backward节点的父级节点是null，则尝试从同名的forward节点寻找父级节点
        """
        # 匹配以.backward.后跟一个或多个数字结尾的模式
        if GraphBuilder.backward_pattern.search(subnode_id) and not upnode_id:
            forward_upnode_id = construct_dict.get(GraphBuilder.backward_pattern.sub(r".forward.\2", subnode_id))
            if forward_upnode_id:
                new_upnode_id = GraphBuilder.forward_pattern.sub(r".backward.\2", forward_upnode_id)
                if new_upnode_id in construct_dict:
                    return new_upnode_id
        # 匹配以.backward结尾的节点
        if subnode_id.endswith(Const.SEP + Const.BACKWARD) and not upnode_id:
            forward_upnode_id = construct_dict.get(subnode_id.replace(Const.BACKWARD, Const.FORWARD))
            if forward_upnode_id:
                new_upnode_id = forward_upnode_id.replace(Const.FORWARD, Const.BACKWARD)
                if new_upnode_id in construct_dict:
                    return new_upnode_id
        return upnode_id

    @staticmethod
    def _init_nodes(graph, construct_dict, data_dict, stack_dict):
        for subnode_id, upnode_id in construct_dict.items():
            upnode_id = GraphBuilder._handle_backward_upnode_missing(construct_dict, subnode_id, upnode_id)
            if upnode_id:
                upnode_op = NodeOp.get_node_op(upnode_id)
                upnode = GraphBuilder._create_or_get_node(graph, [data_dict, stack_dict], upnode_op, upnode_id)
            else:
                upnode = graph.root
            node_op = NodeOp.get_node_op(subnode_id)
            GraphBuilder._create_or_get_node(graph, [data_dict, stack_dict], node_op, subnode_id, upnode)

    @staticmethod
    def _create_or_get_node(graph, data_stack_list, op, name, upnode=None):
        if name in graph.node_map:
            node = graph.get_node(name)
        else:
            graph.add_node(op, name, upnode)
            node = graph.get_node(name)
            node_data = data_stack_list[0].get(name, {})
            node_stack_info = data_stack_list[1].get(name, [])
            # 添加输入输出数据
            input_data, output_data = get_input_output(node_data, node.id)
            # 更新数据
            node.set_input_output(input_data, output_data)
            if GraphConst.BATCH_P2P in name:
                GraphBuilder._extract_batch_p2p_info(node, node_data)
            # 反向节点使用对应前向节点的堆栈信息
            # 模块命名举例：Module.module.module.GPTModel.backward.0; API命名举例：Tensor.permute.1.backward
            if (not node_stack_info and
                    (GraphBuilder.backward_pattern.search(name) or name.endswith(f'{Const.SEP}{Const.BACKWARD}'))):
                forward_node = graph.get_node(
                    # 同名模块全局唯一，无论调用几次堆栈信息都一致，直接使用编号0的同名模块堆栈信息，避免遗漏
                    GraphBuilder.backward_pattern.sub(f'{Const.SEP}{Const.FORWARD}{Const.SEP}0', name)) \
                    if GraphBuilder.backward_pattern.search(name) \
                    else graph.get_node(name.replace(Const.BACKWARD, Const.FORWARD))
                node_stack_info = forward_node.stack_info if forward_node \
                    else ['This backward node cannot find the forward node and cannot retrieve stack information.']
            node.stack_info = node_stack_info
        # 添加节点
        node.add_upnode(upnode)
        return node

    @staticmethod
    def _is_valid_batch_p2p_output(param_list):
        if not isinstance(param_list, list) or not param_list:
            return False
        if not isinstance(param_list[0], list) or not param_list[0]:
            return False
        return True

    @staticmethod
    def _extract_batch_p2p_info(node, node_data):
        param_list = node_data.get(Const.OUTPUT, [])
        # 数据格式："output": [[{param1}, {param2}, ...]]
        if GraphBuilder._is_valid_batch_p2p_output(param_list):
            for param in param_list[0]:
                info = {GraphConst.OP: param.get(GraphConst.OP), GraphConst.PEER: param.get(GraphConst.PEER),
                        GraphConst.GROUP_ID: param.get(GraphConst.GROUP_ID)}
                node.batch_p2p_info.append(info)

    @staticmethod
    def _collect_apis_between_modules(graph):
        """
        图首次展开，这些首层节点包含许多module和api，api数量很多导致图被拉得很长严重影响查阅，因此将module之间的apis收集起来成为节点
        Args:
            graph: 模型结构

        Returns: None
        """
        i = 0
        output = []
        node_list = graph.root.subnodes
        while i < len(node_list):
            current_node = node_list[i]

            # 当前节点为api，检查后续是否还有api
            if current_node.op == NodeOp.function_api:
                temp_nodes = [current_node]
                i += 1
                while i < len(node_list) and node_list[i].op == NodeOp.function_api:
                    temp_nodes.append(node_list[i])
                    i += 1

                # 检查api节点是否大于等于2个
                if len(temp_nodes) >= 2:
                    # 创建新节点，将这些api节点放入新节点的subnodes属性
                    node_id = graph.add_node(NodeOp.api_collection, GraphConst.APIS_BETWEEN_MODULES,
                                             id_accumulation=True)
                    api_collection_node = graph.get_node(node_id)
                    api_collection_node.subnodes = temp_nodes
                    # 重新确立父子关系
                    for node in temp_nodes:
                        node.upnode = api_collection_node
                    api_collection_node.upnode = graph.root
                    output.append(api_collection_node)
                else:
                    # 如果连续的api节点不足2个，将它们原样添加到输出列表
                    output.extend(temp_nodes)
            else:
                # 如果当前节点为module，直接添加到输出列表
                output.append(current_node)
                i += 1

        graph.root.subnodes = output


class GraphExportConfig:
    def __init__(self, graph_n, graph_b=None, tool_tip=None, node_colors=None, micro_steps=None, task='',
                 overflow_check=False):
        self.graph_n = graph_n
        self.graph_b = graph_b
        self.tool_tip = tool_tip
        self.node_colors = node_colors
        self.micro_steps = micro_steps
        self.task = task
        self.overflow_check = overflow_check
