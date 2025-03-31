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

import json
from werkzeug import wrappers, Response, exceptions
from tensorboard.backend import http_util
from ..service.graph_service import GraphService


class GraphView:

    # 获取当前节点对应节点的信息看板数据
    @staticmethod
    @wrappers.Request.application
    def get_node_info(request):
        node_info = json.loads(request.args.get("nodeInfo"))
        meta_data = json.loads(request.args.get("metaData"))
        node_detail = GraphService.get_node_info(node_info, meta_data)
        return http_util.Respond(request, node_detail, "application/json")

    # 添加匹配节点
    @staticmethod
    @wrappers.Request.application
    def add_match_nodes(request):
        npu_node_name = json.loads(request.args.get("npuNodeName"))
        bench_node_name = json.loads(request.args.get("benchNodeName"))
        meta_data = json.loads(request.args.get("metaData"))
        match_result = GraphService.add_match_nodes(npu_node_name, bench_node_name, meta_data)
        return http_util.Respond(request, match_result, "application/json")

    # 取消节点匹配
    @staticmethod
    @wrappers.Request.application
    def delete_match_nodes(request):
        npu_node_name = json.loads(request.args.get("npuNodeName"))
        bench_node_name = json.loads(request.args.get("benchNodeName"))
        meta_data = json.loads(request.args.get("metaData"))
        match_result = GraphService.delete_match_nodes(npu_node_name, bench_node_name, meta_data)
        return http_util.Respond(request, match_result, "application/json")

    # 获取匹配节点列表
    @staticmethod
    @wrappers.Request.application
    def get_matched_state_list(request):
        meta_data = json.loads(request.args.get("metaData"))
        matched_state_list = GraphService.get_matched_state_list(meta_data)
        return http_util.Respond(request, matched_state_list, "application/json")

    # 保存匹配节点列表
    @staticmethod
    @wrappers.Request.application
    def save_data(request):
        meta_data = json.loads(request.args.get("metaData"))
        save_result = GraphService.save_data(meta_data)
        return http_util.Respond(request, save_result, "application/json")
    
    # 更新颜色信息
    @staticmethod
    @wrappers.Request.application
    def update_colors(request):
        run = json.loads(request.args.get('run'))
        colors = json.loads(request.args.get('colors'))
        update_result = GraphService.update_colors(run, colors)
        return http_util.Respond(request, update_result, "application/json")
