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
from werkzeug import wrappers
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.util import tb_logging

from . import constants
from .app.views.graph_views import GraphView
from .app.utils.graph_utils import GraphUtils
from .app.utils.global_state import GraphState

logger = tb_logging.get_logger()

PLUGIN_NAME = 'graph_ascend'
PLUGIN_NAME_RUN_METADATA_WITH_GRAPH = 'graph_ascend_run_metadata_graph'


class GraphsPlugin(base_plugin.TBPlugin):
    """Graphs Plugin for TensorBoard."""

    plugin_name = PLUGIN_NAME
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
        GraphState.set_global_value('logdir', os.path.abspath(os.path.expanduser(context.logdir.rstrip('/'))))
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
            '/index.js': GraphView.static_file_route,
            '/index.html': GraphView.static_file_route,
            "/load_meta_dir": GraphView.load_meta_dir,
            "/screen": self.get_all_screen_nodes,
            '/loadGraphData': GraphView.load_graph_data,
            '/loadGraphConfigInfo': GraphView.load_graph_config_info,
            '/loadGraphAllNodeList': GraphView.load_graph_all_node_list,
            '/changeNodeExpandState': GraphView.change_node_expand_state,
            '/updateHierarchyData': GraphView.update_hierarchy_data,
            '/getNodeInfo': GraphView.get_node_info,
            '/addMatchNodes': GraphView.add_match_nodes,
            '/addMatchNodesByConfig': GraphView.add_match_nodes_by_config,
            '/deleteMatchNodes': GraphView.delete_match_nodes,
            '/saveData': GraphView.save_data,
            '/updateColors': GraphView.update_colors,
            '/saveMatchedRelations': GraphView.save_matched_relations,
        }

    def is_active(self):
        """The graphs plugin is active if any run has a graph."""
        for content in os.listdir(self.logdir):
            content_path = os.path.join(self.logdir, content)
            if os.path.isfile(content_path) and content.endswith('.vis'):
                return True
            if os.path.isdir(content_path):
                for file in os.listdir(content_path):
                    if os.path.isfile(os.path.join(content_path, file)) and file.endswith('.vis'):
                        return True
        return False

    def data_plugin_names(self):
        return (
            PLUGIN_NAME,
            PLUGIN_NAME_RUN_METADATA_WITH_GRAPH,
        )

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            es_module_path='/index.js',
            disable_reload=True,
        )

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

    def dfs_collect_nodes(self, json_data, request):
        root_subnodes_set = []
        all_node_names = []
        try:
            request_micro_step_id = request.args.get("microStep")
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

    # 检查到底是读一般还是用之前存的
    def check_jsondata(self, request):
        meta_data = {
            "tag": request.args.get("tag"),
            "run": request.args.get('run')
        }
        graph_data, _ = GraphUtils.get_graph_data(meta_data)
        return graph_data

    def json_get(self, data, *args):
        result = data
        for key in args:
            if result is None:
                return None
            result = result.get(key)
        return result
