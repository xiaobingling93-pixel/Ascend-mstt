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
import json
from pathlib import Path
from werkzeug import wrappers, Response, exceptions
from tensorboard.backend import http_util
from ..service.graph_service import GraphService
from ..utils.graph_utils import GraphUtils


class GraphView:

    # 静态文件路由
    @staticmethod
    @wrappers.Request.application
    def static_file_route(request):
        filename = os.path.basename(request.path)
        extension = os.path.splitext(filename)[1]
        if extension == '.html':
            mimetype = 'text/html'
        elif extension == '.js':
            mimetype = 'application/javascript'
        else:
            mimetype = 'application/octet-stream'
        current_dir = Path(__file__).resolve().parent
        server_dir = current_dir.parent.parent
        filepath = server_dir / "static" / filename
        try:
            with open(filepath, 'rb') as infile:
                contents = infile.read()
        except IOError as e:
            raise exceptions.NotFound('404 Not Found') from e
        return Response(contents, content_type=mimetype, headers={"X-Content-Type-Options": "nosniff"})

    @staticmethod
    @wrappers.Request.application
    def load_meta_dir(request):
        """Scan logdir for directories containing .vis files, modified to return a tuple of (run, tag)."""
        is_safe_check = GraphUtils.safe_json_loads(request.args.get("isSafeCheck"))
        result = GraphService.load_meta_dir(is_safe_check)
        response = http_util.Respond(request, result, "application/json")
        return response

    # 读取当前图数据
    @staticmethod
    @wrappers.Request.application
    def load_graph_data(request):
        run = request.args.get("run")
        tag = request.args.get("tag")
        return Response(
            GraphService.load_graph_data(run, tag),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'close',  # TCP链接不复用，请求结束释放资源
                "X-Content-Type-Options": "nosniff",
            }
        )

    # 获取当前图数据配置信息
    @staticmethod
    @wrappers.Request.application
    def load_graph_config_info(request):
        run = request.args.get("run")
        tag = request.args.get("tag")
        result = GraphService.load_graph_config_info(run, tag)
        # 创建响应对象
        response = http_util.Respond(request, result, "application/json")
        return response

    # 获取当前图所有节点列表
    @staticmethod
    @wrappers.Request.application
    def load_graph_all_node_list(request):
        run = request.args.get("run")
        tag = request.args.get("tag")
        micro_step = request.args.get('microStep')
        result = GraphService.load_graph_all_node_list(run, tag, micro_step)
        response = http_util.Respond(request, result, "application/json")
        return response

    # 展开关闭节点
    @staticmethod
    @wrappers.Request.application
    def change_node_expand_state(request):
        node_info = GraphUtils.safe_json_loads(request.args.get("nodeInfo"))
        meta_data = GraphUtils.safe_json_loads(request.args.get("metaData"))
        hierarchy = GraphService.change_node_expand_state(node_info, meta_data)
        return http_util.Respond(request, json.dumps(hierarchy), "application/json")

    # 更新当前图节点信息
    @staticmethod
    @wrappers.Request.application
    def update_hierarchy_data(request):
        graph_type = request.args.get("graphType")
        hierarchy = GraphService.update_hierarchy_data(graph_type)
        return http_util.Respond(request, json.dumps(hierarchy), "application/json")

    # 获取当前节点对应节点的信息看板数据
    @staticmethod
    @wrappers.Request.application
    def get_node_info(request):
        node_info = GraphUtils.safe_json_loads(request.args.get("nodeInfo"))
        meta_data = GraphUtils.safe_json_loads(request.args.get("metaData"))
        node_detail = GraphService.get_node_info(node_info, meta_data)
        return http_util.Respond(request, json.dumps(node_detail), "application/json")

    # 根据配置文件添加匹配节点
    @staticmethod
    @wrappers.Request.application
    def add_match_nodes_by_config(request):
        config_file = request.args.get("configFile")
        meta_data = GraphUtils.safe_json_loads(request.args.get("metaData"))
        match_result = GraphService.add_match_nodes_by_config(config_file, meta_data)
        return http_util.Respond(request, json.dumps(match_result), "application/json")

    # 添加匹配节点
    @staticmethod
    @wrappers.Request.application
    def add_match_nodes(request):
        npu_node_name = request.args.get("npuNodeName")
        bench_node_name = request.args.get("benchNodeName")
        meta_data = GraphUtils.safe_json_loads(request.args.get("metaData"))
        is_match_children = GraphUtils.safe_json_loads(request.args.get("isMatchChildren"))
        match_result = GraphService.add_match_nodes(npu_node_name, bench_node_name, meta_data, is_match_children)
        return http_util.Respond(request, json.dumps(match_result), "application/json")

    # 取消节点匹配
    @staticmethod
    @wrappers.Request.application
    def delete_match_nodes(request):
        npu_node_name = request.args.get("npuNodeName")
        bench_node_name = request.args.get("benchNodeName")
        meta_data = GraphUtils.safe_json_loads(request.args.get("metaData"))
        is_unmatch_children = GraphUtils.safe_json_loads(request.args.get("isUnMatchChildren"))
        match_result = GraphService.delete_match_nodes(npu_node_name, bench_node_name, meta_data, is_unmatch_children)
        return http_util.Respond(request, json.dumps(match_result), "application/json")

    # 保存匹配节点列表
    @staticmethod
    @wrappers.Request.application
    def save_data(request):
        meta_data = GraphUtils.safe_json_loads(request.args.get("metaData"))
        save_result = GraphService.save_data(meta_data)
        return http_util.Respond(request, json.dumps(save_result), "application/json")

    # 更新颜色信息
    @staticmethod
    @wrappers.Request.application
    def update_colors(request):
        run = request.args.get('run')
        colors = GraphUtils.safe_json_loads(request.args.get('colors'))
        update_result = GraphService.update_colors(run, colors)
        return http_util.Respond(request, json.dumps(update_result), "application/json")

    # 保存匹配关系
    @staticmethod
    @wrappers.Request.application
    def save_matched_relations(request):
        meta_data = GraphUtils.safe_json_loads(request.args.get("metaData"))
        save_result = GraphService.save_matched_relations(meta_data)
        return http_util.Respond(request, json.dumps(save_result), "application/json")
