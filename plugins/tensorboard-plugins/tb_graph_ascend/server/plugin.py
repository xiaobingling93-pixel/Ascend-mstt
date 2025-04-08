# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#
# Copyright (c) 2025, Huawei Technologies.
# Adapt to the model hierarchical visualization data collected by the msprobe tool
# ==============================================================================
"""The TensorBoard Graphs plugin."""

import os
from werkzeug import wrappers, Response, exceptions
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.util import tb_logging

from . import constants
from .app.views.graph_views import GraphView
from .app.utils.graph_utils import GraphUtils
from .app.utils.global_state import set_global_value, get_global_value

logger = tb_logging.get_logger()


class GraphsPlugin(base_plugin.TBPlugin):
    """Graphs Plugin for TensorBoard."""

    plugin_name = constants.PLUGIN_NAME
    headers = [('X-Content-Type-Options', 'nosniff')]

    def __init__(self, context):
        """Instantiates GraphsPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        super().__init__(context)
        self._data_provider = context.data_provider
        self.logdir = os.path.abspath(os.path.expanduser(context.logdir.rstrip('/')))
        # 将logdir赋值给global_state中的logdir属性,方便其他模块使用
        set_global_value('logdir', os.path.abspath(os.path.expanduser(context.logdir.rstrip('/'))))
        self._current_file_path = None  # Store the path of the currently loaded file
        self._current_file_data = None  # Store the data of the currently loaded file
        self._current_tag = None  # Store the tag of the currently loaded file
        self.batch_id = 0  # 将 batch_id 声明为实例变量
        self.step_id = 0  # 可以同样声明 step_id
        self.dfs_node_ids = []  # batch和step没变的话就将所有的nodename存起来，方便快速读取
        self.check_batch_id = -1  # 来配合node_ids监察用的，他不变node_ids就不用重新读取了
        self.check_step_id = 0  # 同上
        self.check_tag = None

    def get_plugin_apps(self):
        return {
            '/index.js': self.static_file_route,
            '/index.html': self.static_file_route,
            "/info": self.info_route,
            "/components": self.get_all_data,
            "/expandnodes": self.get_all_upnodes,
            "/screen": self.get_all_screen_nodes,
            "/parent": self.get_parent_node,
            "/rank": self.get_rank,
            "/subgraph": self.subgraph_route,
            '/getNodeInfo': GraphView.get_node_info,
            '/addMatchNodes': GraphView.add_match_nodes,
            '/deleteMatchNodes': GraphView.delete_match_nodes,
            '/getMatchedStateList': GraphView.get_matched_state_list,
            '/saveData': GraphView.save_data,
            '/updateColors': GraphView.update_colors,
        }

    def is_active(self):
        """The graphs plugin is active iff any run has a graph."""
        for _, _, files in os.walk(self.logdir):
            for file in files:
                if file.endswith('.vis'):
                    return True
        return False

    def data_plugin_names(self):
        return (
            constants.PLUGIN_NAME,
            constants.PLUGIN_NAME_RUN_METADATA_WITH_GRAPH,
        )

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            es_module_path='/index.js',
            disable_reload=True,
        )

    def info_impl(self):
        """Returns a dict of all runs and their data availabilities"""
        result = {}

        def add_row_item(run, tag=None, is_vis=False):
            run_item = result.setdefault(
                run,
                {"run": run, "tags": {}},
            )

            tag_item = None
            if tag:
                tag_item = run_item.get("tags").setdefault(
                    tag,
                    {
                        "tag": tag,
                        "conceptual_graph": False,
                        "op_graph": False,
                        "profile": False,
                    },
                )
            return (run_item, tag_item)

        run_tag_pairs = self._get_run_dirs()
        for run, tag in run_tag_pairs:
            try:
                add_row_item(run, tag, is_vis=True)
            except ValueError as e:
                logger.error(f"Warning: Skipping invalid run/tag pair ({run}, {tag}). Error: {e}")
              
        return result

    # 拿所有nodename的
    def get_all_node_names(self, json_data, request):
        npu_ids, bench_ids = [], []
        batch = request.args.get("batch")
        step = request.args.get("step")
        if batch is None or step is None:
            logger.error('The param "batch" or "step" does not exist or not a valid value')
        # 获取 NPU 和 Bench 数据
        npu_data = self.json_get(json_data, 'NPU')
        bench_data = self.json_get(json_data, 'Bench')

        def extract_ids(nodes_data, id_list):
            for node_name in nodes_data.get("node"):
                id_list.append(node_name)

        def traverse_npu(subnodes):
            for node in subnodes:
                node_data = (
                    self.json_get(npu_data, 'node', node) if npu_data else self.json_get(json_data, 'node', node)
                )
                micro_step_id = node_data.get('micro_step_id')
                if str(micro_step_id) == batch or micro_step_id is None:
                    npu_ids.append(node)
                    traverse_npu(node_data.get('subnodes', []))

        # 提取 NPU 节点 ID
        if batch == '-1' and step == '-1':
            extract_ids(npu_data or json_data, npu_ids)
        else:
            root = (npu_data or json_data).get('root')
            root_subnodes = self.json_get((npu_data or json_data), 'node', root, 'subnodes')
            traverse_npu(root_subnodes)

        # 提取 Bench 节点 ID
        if bench_data:
            extract_ids(bench_data, bench_ids)
        # 返回格式为 [[NPU节点ID列表], [Bench节点ID列表]]
        return [npu_ids, bench_ids]

    def dfs_collect_nodes(self, json_data, request): 
        root_subnodes_set = []
        all_node_names = []
        try:
            request_micro_step_id = request.args.get("batch")
        except ValueError:
            logger.error('The param "batch" or "step" does not exist or not a valid value')
        root_name = self.json_get(json_data, 'NPU', 'root') or \
                    self.json_get(json_data, 'root')
        root_subnodes = self.json_get(json_data, 'NPU', 'node', root_name, 'subnodes') \
                        if 'NPU' in json_data else \
                        self.json_get(json_data, 'node', root_name, 'subnodes')
        if root_subnodes:
            for node in root_subnodes:
                json_path = ['NPU', 'node', node, 'micro_step_id'] if 'NPU' in json_data \
                    else ['node', node, 'micro_step_id']   
                micro_step_id = self.json_get(json_data, *json_path)
                if request_micro_step_id == '-1' or str(micro_step_id) == request_micro_step_id:
                    root_subnodes_set.append(node)
        
        def get_leaf_nodes(subnodes_set):
            npu_data = self.json_get(json_data, 'NPU')
            for node in subnodes_set:
                node_data = (
                    self.json_get(npu_data, 'node', node) if npu_data else self.json_get(json_data, 'node', node)
                )
                if node_data:
                    if node_data.get('subnodes'):
                        get_leaf_nodes(node_data.get('subnodes'))
                    else:
                        all_node_names.append(node)
        
        get_leaf_nodes(root_subnodes_set)
        
        return all_node_names

    # 拿所有precisonNodes的，与controls的精度筛选联动
    @wrappers.Request.application
    def get_all_screen_nodes(self, request):
        grouped_screen_set, inaccuracy_node_ids = [], []
        precision_none = 0
        screen = ''
        # 尝试获取 screen_set 和 screen 的值
        for key, value in constants.SCREEN_MAP.items():
            if key in request.args:
                screen_set = request.args.get(key)
                screen = value
                break  # 找到一个匹配的 key 后跳出循环

        if screen == 'precision_index':
            precision_set_str = screen_set.split(',')
            if constants.UNMATCHED_NODE_NAME in precision_set_str:
                precision_set_str = [p for p in precision_set_str if p != constants.UNMATCHED_NODE_NAME]
                precision_none = 1
            grouped_screen_set = [
                list(map(float, precision_set_str[i: i + 2])) 
                for i in range(0, len(precision_set_str), 2)
            ]
        else:
            grouped_screen_set = screen_set
        tag = request.args.get("tag")
        json_data = self.check_jsondata(request)

        def has_conditions_changed(tag, batch):
            return (
                self.check_batch_id != batch
                or self.check_step_id != self.step_id
                or self.check_tag != tag
                or self.check_tag is None
            )

        if has_conditions_changed(tag, self.batch_id):
            self.dfs_node_ids = self.dfs_collect_nodes(json_data, request)
            self.check_batch_id = self.batch_id
            self.check_step_id = self.step_id
            self.check_tag = tag
        node_ids = self.dfs_node_ids
        for node in node_ids:
            node_data = self.json_get(json_data, 'NPU', 'node', node, 'data') or self.json_get(
                json_data, 'node', node, 'data'
            )
            matched = self.json_get(json_data, 'NPU', 'node', node, 'matched_node_link') or self.json_get(
                json_data, 'node', node, 'matched_node_link'
            )
            inaccuracy = node_data.get(screen) if node_data is not None else None
            # 如果 inaccuracy 为 None，直接检查是否符合条件
            if inaccuracy is None and precision_none == 0: 
                continue  # 跳过后续的处理，进入下一个 node
            if inaccuracy is None and precision_none == 1:
                if (node_data is None or node_data.get('overflow_level', False)) and not matched:
                    inaccuracy_node_ids.append(node)
                continue  # 跳过后续的处理，进入下一个 node

            # 对于 inaccuracy 是数字类型，检查是否在某个子范围内，精度误差
            if isinstance(inaccuracy, (int, float)):
                for group in grouped_screen_set:
                    if len(group) > 1 and all(g is not None for g in group) and group[0] <= inaccuracy <= group[1]:
                        inaccuracy_node_ids.append(node)
                        break  # 找到符合条件的，跳出当前循环
            # 对于非数字的 inaccuracy，检查是否在 grouped_screen_set 中，溢出检测
            elif inaccuracy in grouped_screen_set:
                inaccuracy_node_ids.append(node)
            else:
                logger.error(f'The inaccuracy in {node} is not a valid value')

        return http_util.Respond(request, inaccuracy_node_ids, "application/json")

    def group_precision_set(self, precision_set):
        if len(precision_set) % 2 != 0:
            raise ValueError('The number of elements in precision_set is not even')
        grouped_precision_set = [precision_set[i: i + 2] for i in range(0, len(precision_set), 2)]
        return grouped_precision_set

    def get_all_unmatched_nodes(self, all_node_names, json_data):
        is_npu_present = 'NPU' in json_data

        def collect_unmatched_nodes(node_list, *path):
            return [node for node in node_list if not self.json_get(json_data, *path, node, 'matched_node_link')]

        npu_unmatched = (
            collect_unmatched_nodes(all_node_names[0], 'NPU', 'node')
            if is_npu_present
            else collect_unmatched_nodes(all_node_names[0], 'node')
        )
        bench_unmatched = collect_unmatched_nodes(all_node_names[1], 'Bench', 'node') if is_npu_present else []
        return [npu_unmatched, bench_unmatched]

    @wrappers.Request.application
    def get_parent_node(self, request):
        node = request.args.get("node")[4:]  # 获取节点信息
        prefix = request.args.get("node")[:4]  # 获取前缀
        json_data = self.check_jsondata(request)  # 检查请求中的 JSON 数据

        def find_upnode(node):
            matched_node_link_list = self.json_get(
                json_data, constants.PREFIX_MAP[prefix], 'node', node, 'matched_node_link'
            )

            if matched_node_link_list:
                result = matched_node_link_list[-1]  # 获取匹配的最后一个节点
                return http_util.Respond(request, result, "application/json")  # 返回响应
            else:
                # 如果没有找到 matched_node_link，继续递归查找上级节点
                upnode = self.json_get(json_data, constants.PREFIX_MAP[prefix], 'node', node, 'upnode')
                if upnode:
                    return find_upnode(upnode)  # 递归查找上级节点
                else:
                    return http_util.Respond(request, {}, "application/json")  # 如果没有找到上级节点，返回空响应

        return find_upnode(node)

    # 多卡跳转后端的方法，负责拿到rank数据
    @wrappers.Request.application
    def get_rank(self, request):
        node_name = request.args.get('node')
        side = request.args.get('side')

        # 构建 JSON 路径
        json_path = [side] if side else []  # 如果 side 存在，路径中包含 side
        json_path.extend(['node', node_name, 'matched_distributed'])

        # 获取 matched_distributed
        matched_distributed = self.json_get(self._current_file_data, *json_path)

        # 返回结果
        if matched_distributed:
            return http_util.Respond(request, matched_distributed, "application/json")
        else:
            return http_util.Respond(request, {}, "application/json")

    # 拿json_data里面所有配置数据的
    @wrappers.Request.application
    def get_all_data(self, request):
        """Returns all data in json format."""
        response_data = {}
        tag = request.args.get('tag')
        run = request.args.get('run')
        json_data = self.check_jsondata(request)
        if not json_data:
            return http_util.Respond(
                request, f"Failed to check file '{tag}', view the log for detail.", "text/plain", 400
            )
        self._current_file_data = json_data
        if json_data.get('StepList', {}) and 'ALL' not in json_data.get('StepList', {}):
            json_data['StepList'].insert(0, 'ALL')
        all_node_names = self.get_all_node_names(json_data, request)

        # 读取第一个文件中的Colors和OverflowCheck
        first_run_tag = get_global_value("first_run_tag")
        first_file_data, _ = GraphUtils.safe_load_data(run, first_run_tag)
        # 读取全局信息
        response_data['menu'] = all_node_names
        response_data['unMatchedNode'] = self.get_all_unmatched_nodes(all_node_names, json_data)
        if json_data.get('MicroSteps', {}):
            response_data['microSteps'] = json_data.get('MicroSteps')
        if json_data.get('StepList', {}):
            response_data['stepList'] = json_data.get('StepList')
        if json_data.get('Tooltips', {}):
            response_data['tooltips'] = json_data.get('Tooltips')
        if 'Colors' in first_file_data:
            response_data['colors'] = first_file_data.get('Colors')
        if 'OverflowCheck' in first_file_data:
            response_data['overflowCheck'] = first_file_data.get('OverflowCheck')
        else:
            # 适配老精度溢出数据
            response_data['overflowCheck'] = True
        return http_util.Respond(request, response_data, "application/json")

    @wrappers.Request.application
    def static_file_route(self, request):
        filename = os.path.basename(request.path)
        extension = os.path.splitext(filename)[1]
        if extension == '.html':
            mimetype = 'text/html'
        elif extension == '.js':
            mimetype = 'application/javascript'
        else:
            mimetype = 'application/octet-stream'
        filepath = os.path.join(os.path.dirname(__file__), 'static', filename)
        try:
            with open(filepath, 'rb') as infile:
                contents = infile.read()
        except IOError as e:
            raise exceptions.NotFound('404 Not Found') from e
        return Response(contents, content_type=mimetype, headers=GraphsPlugin.headers)

    # 方便多层级展开的upnodes节点集合，与tf-graph的_menuSelectedNodeExpand联动
    @wrappers.Request.application
    def get_all_upnodes(self, request):
        npu_upnodes_list, matched_upnodes_list, node_list = [], [], []
        node, matched_node, prefix = '', '', ''
        node_arg = request.args.get('node')
        json_data = self.check_jsondata(request)
        prefix = str(node_arg)[:4] if str(node_arg)[:4] in constants.PREFIX_MAP else ''
        node = node_arg[4:] if prefix in constants.PREFIX_MAP else node_arg
        if prefix in constants.PREFIX_MAP and json_data.get(constants.PREFIX_MAP[prefix], {}):
            node_list = json_data[constants.PREFIX_MAP[prefix]].get('node', {})
        else:
            node_list = json_data.get('node', {})
        matched_node = (
            node_list.get(node, {}).get('matched_node_link', [])[-1]
            if node_list.get(node, {}).get('matched_node_link')
            else None
        )

        def get_upnodes(node, prefix):
            upnodes_list = []
            if prefix == '':
                node_list = json_data.get('node', {})
            else:
                node_list = json_data.get('NPU' if prefix == 'N___' else 'Bench', {}).get('node', {})
            while node in node_list:
                upnode = node_list[node].get('upnode')
                if not upnode or upnode == 'None':
                    break
                upnodes_list.insert(0, upnode)
                node = upnode
            return upnodes_list

        npu_upnodes_list = get_upnodes(node, prefix)
        # 如果 matched_node 是 None 的话
        if matched_node is None:
            previous_node = None  # 用于跟踪上一个 node
            for node in reversed(npu_upnodes_list):
                if node_list.get(node, {}).get('matched_node_link'):  # 判断条件
                    matched_node = previous_node  # 将 matched_node 设置为上一个 node
                    break
                previous_node = node  # 更新 previous_node 为当前 node
        if prefix in constants.PREFIX_MAP:
            matched_upnodes_list = get_upnodes(matched_node, prefix)
        return http_util.Respond(request, [[prefix], npu_upnodes_list, matched_upnodes_list], "application/json")

    # 检查到底是读一般还是用之前存的
    def check_jsondata(self, request):
        meta_data = {
            "tag": request.args.get("tag"),
            "run": request.args.get('run')
        }
        graph_data, _ = GraphUtils.get_graph_data(meta_data)
        return graph_data

    # 处理xx.get
    def json_get(self, data, *args):
        result = data
        for key in args:
            if result is None:
                return None
            result = result.get(key)
        return result

    # 获取子图数据，最核心且基本的所在
    @wrappers.Request.application
    def subgraph_route(self, request):
        """Returns a subgraph for a given node id, modified to use run and tag from query parameters."""
        json_data = self.check_jsondata(request)
        if not json_data:
            return http_util.Respond(request, "Failed to get subgraph, view the log for details.", "text/plain", 400)
        node_id = request.args.get("node")
        self.batch_id = request.args.get("batch")
        self.step_id = request.args.get("step")
        if node_id is None:
            return http_util.Respond(request, 'The query parameter "node" is required', "text/plain", 400)
        if node_id == 'root':
            if json_data.get('Bench', {}):
                subgraph_pbtxt_set = {}
                for node_type in ('Bench', 'NPU'):
                    subgraph = {'node': {}, 'edge': {}}
                    node = self.json_get(json_data, constants.SETS[node_type][0], 'root')
                    node_data = self.json_get(json_data, constants.SETS[node_type][0], 'node', node)
                    node = constants.SETS[node_type][1] + node
                    matched_node_link = node_data['matched_node_link']
                    if matched_node_link[0][:4] != constants.SETS[node_type][2]:
                        matched_node_link[0] = constants.SETS[node_type][2] + matched_node_link[0]
                    subgraph['node'][node] = node_data
                    subgraph_pbtxt_set[node_type] = self._convert_to_protobuf_format(subgraph)
                subgraph_pbtxt = subgraph_pbtxt_set.get('NPU', '') + subgraph_pbtxt_set.get('Bench', '')
            else:
                subgraph = {'node': {}, 'edge': {}}
                node = json_data.get('root')
                node_data = self.json_get(json_data, 'node', node)
                subgraph['node'][node] = node_data
                subgraph_pbtxt = self._convert_to_protobuf_format(subgraph)
        else:
            subgraph = self._extract_subgraph(json_data, node_id)
            subgraph_pbtxt = self._convert_to_protobuf_format(subgraph)
        return http_util.Respond(request, subgraph_pbtxt, "text/x-protobuf")

    @wrappers.Request.application
    def info_route(self, request):
        info = self.info_impl()
        return http_util.Respond(request, info, "application/json")

    # 同上二者一体
    def _extract_subgraph(self, json_data, node_id):
        """提取子图，支持多种节点前缀逻辑"""
        subgraph = {'node': {}, 'edge': []}

        # 检查前缀并获取节点集合
        prefix = node_id[:4]
        if prefix in constants.SETS and len(prefix) == 4:
            node_id = node_id[4:]
            node_set = self.json_get(json_data, constants.SETS[prefix][0], 'node')
        else:
            prefix = ''
            node_set = json_data.get('node', {})

        # 获取当前节点数据
        node_data = node_set.get(node_id, {})
        subnodes = node_data.get('subnodes', [])

        # 遍历子节点
        for subnode_id in subnodes:
            subnode_id_data = node_set.get(subnode_id, {})
            if subnode_id_data.get('micro_step_id') is not None:
                self._process_subnode(subgraph, prefix, subnode_id, subnode_id_data, json_data)
            else:
                self._process_non_root_subnode(subgraph, prefix, subnode_id, subnode_id_data)

        return subgraph

    def _process_non_root_subnode(self, subgraph, prefix, subnode_id, subnode_id_data):
        """处理非根子节点"""
        # 更新匹配的节点链接
        self._update_matched_node_links(subnode_id_data, prefix)

        # 添加前缀并存入子图
        full_subnode_id = prefix + subnode_id
        subgraph['node'][full_subnode_id] = subnode_id_data

    # 针对分micro_step_id和step_id取的部分节点
    def _process_subnode(self, subgraph, prefix, subnode_id, subnode_id_data, json_data):
        batchid = subnode_id_data.get('micro_step_id')
        stepid = subnode_id_data.get('step_id')
        steplist = json_data.get('StepList')

        def should_update_node():
            """判断是否需要更新节点的条件逻辑"""
            if self.batch_id == '-1':
                if self.step_id == '-1':  # batch_id 和 step_id 都为 -1
                    return True
                return stepid == str(steplist[int(self.step_id) + 1])  # 匹配 step_id
            else:  # batch_id 有效
                if self.step_id != '-1':  # step_id 有效
                    return batchid == int(self.batch_id) and stepid == str(steplist[int(self.step_id) + 1])
                return batchid == int(self.batch_id)  # 仅匹配 batch_id

        if should_update_node():
            self._update_matched_node_links(subnode_id_data, prefix)
            subnode_id = prefix + subnode_id
            subgraph['node'][subnode_id] = subnode_id_data

    def _update_matched_node_links(self, subnode_id_data, prefix):
        if 'matched_node_link' in subnode_id_data:
            for index, matched_node_link in enumerate(subnode_id_data['matched_node_link']):
                if matched_node_link[:4] != constants.SETS[prefix][1]:
                    matched_node_link = constants.SETS[prefix][1] + matched_node_link
                subnode_id_data['matched_node_link'][index] = matched_node_link

    # 拼接成类json
    def _convert_to_protobuf_format(self, subgraph):
        """Converts subgraph data to the protobuf text format expected by the frontend."""
        nodes = subgraph.get('node', {})
        protobuf_format = ""
        for node_id, node_data in nodes.items():
            protobuf_format += f'node {{\n  name: "{node_id}"\n  op: "{node_data.get("id")}"\n'
            protobuf_format += f'  node_type: {node_data.get("node_type", 0)}\n'
            if node_data.get("matched_node_link"):
                protobuf_format += f'  matched_node_link: {node_data.get("matched_node_link")}\n'
            protobuf_format += f'  attr: "{node_data.get("data", "{}")}"\n'.replace('True', 'true').replace(
                'False', 'false'
            )
            protobuf_format += f'  precision_index: {(node_data.get("data", {}).get("precision_index"))}\n'
            if node_data.get("input_data"):
                protobuf_format += f'  input_data: "{node_data.get("input_data", "{}")}"\n'
            if node_data.get("output_data"):
                protobuf_format += f'  output_data: "{node_data.get("output_data", "{}")}"\n'
            protobuf_format += f'  suggestions: "{node_data.get("suggestions", "{}")}"\n'
            if not node_data.get("subnodes"):
                protobuf_format += f'  isLeaf: true\n'
            else:
                protobuf_format += f'  isLeaf: false\n'
                protobuf_format += f'  subnodes: {node_data.get("subnodes")}\n'
            if node_data.get("stack_info"):
                protobuf_format += f'  stack_info: {node_data.get("stack_info")}\n'
            protobuf_format += '}\n'
        return protobuf_format

    def _get_run_dirs(self):
        """Scan logdir for directories containing .vis files, modified to return a tuple of (run, tag)."""
        run_tag_pairs = []
        for root, _, files in os.walk(self.logdir):
            for file in files:
                if file.endswith('.vis'):  # check for .vis extension
                    run = os.path.abspath(root)
                    tag = os.path.splitext(file)[0]  # Use the filename without extension as tag
                    _, error = GraphUtils.safe_load_data(run, tag, True)
                    if error:
                        logger.error(f'Error: File run:"{run},tag:{tag}" is not accessible. Error: {error}')
                        continue
                    run_tag_pairs.append((run, tag))
        if len(run_tag_pairs) > 0:
            set_global_value('first_run_tag', run_tag_pairs[0][1])
        return run_tag_pairs

