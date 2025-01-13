# Copyright (c) 2024, Huawei Technologies Co., Ltd.
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
from msprobe.visualization.builder.msprobe_adapter import compare_node, get_compare_mode, run_real_data
from msprobe.visualization.utils import GraphConst, load_json_file, load_data_json_file, get_csv_df
from msprobe.visualization.graph.graph import Graph, NodeOp
from msprobe.visualization.graph.node_colors import NodeColors
from msprobe.visualization.compare.mode_adapter import ModeAdapter
from msprobe.core.common.const import Const


class GraphComparator:
    def __init__(self, graphs, dump_path_param, args, mapping_dict=None):
        self.graph_n = graphs[0]
        self.graph_b = graphs[1]
        self._parse_param(dump_path_param, args.output_path)
        self.framework = args.framework
        self.mapping_dict = mapping_dict
        self.fuzzy_match = args.fuzzy_match
        self.pattern = re.compile(r'\.\d+\.')

    def compare(self):
        """
        比较函数，初始化结束后单独调用。比较结果写入graph_n
        """
        if self.fuzzy_match:
            self._compare_nodes_fuzzy(self.graph_n.root)
        else:
            self._compare_nodes(self.graph_n.root)
        self._postcompare()
    
    def add_compare_result_to_node(self, node, compare_result_list):
        """
        将比对结果添加到节点的输入输出数据中
        Args:
            node: 节点
            compare_result_list: 包含参数信息和对比指标（真实数据对比模式除外）的list
        """
        # 真实数据比对，先暂存节点，在多进程对比得到精度指标后，再将指标添加到节点中
        if self.ma.prepare_real_data(node):
            return
        compare_in_dict = {}
        compare_out_dict = {}
        # input和output对比数据分开
        for item in compare_result_list:
            if not isinstance(item, (list, tuple)) or not item:
                continue
            if '.output.' in item[0]:
                compare_out_dict[item[0]] = item
            else:
                compare_in_dict[item[0]] = item
        precision_index, other_dict = (
            self.ma.parse_result(node, [compare_in_dict, compare_out_dict]))
        node.data[GraphConst.JSON_INDEX_KEY] = precision_index
        node.data.update(other_dict)
    
    def _parse_param(self, dump_path_param, output_path):
        self.dump_path_param = dump_path_param
        self.output_path = output_path
        compare_mode = get_compare_mode(self.dump_path_param)
        self.ma = ModeAdapter(compare_mode)
        self.data_n_dict = load_data_json_file(dump_path_param.get('npu_json_path'))
        self.data_b_dict = load_data_json_file(dump_path_param.get('bench_json_path'))
        self.stack_json_data = load_json_file(dump_path_param.get('stack_json_path'))

    def _postcompare(self):
        self._handle_api_collection_index()
        if not self.ma.compare_mode == GraphConst.REAL_DATA_COMPARE:
            return
        df = get_csv_df(True, self.ma.csv_data, self.ma.compare_mode)
        df = run_real_data(self.dump_path_param, df, self.framework, True if self.mapping_dict else False)
        compare_data_dict = {row[0]: row.tolist() for _, row in df.iterrows()}
        for node in self.ma.compare_nodes:
            precision_index, _ = self.ma.parse_result(node, [compare_data_dict])
            node.data[GraphConst.JSON_INDEX_KEY] = precision_index

    def _handle_api_collection_index(self):
        """
        api集合的指标, md5模式使用集合中所有api最小的指标，statistics和tensor模式使用集合中所有api最大的指标
        md5模式下指标为0代表最差，statistics和tensor模式下指标为1代表最差
        """
        for node in self.graph_n.root.subnodes:
            if node.op == NodeOp.api_collection:
                precision_index = GraphConst.MAX_INDEX_KEY if self.ma.compare_mode == GraphConst.MD5_COMPARE \
                    else GraphConst.MIN_INDEX_KEY
                for api in node.subnodes:
                    precision_index = min(precision_index,
                                          api.data.get(GraphConst.JSON_INDEX_KEY, GraphConst.MAX_INDEX_KEY)) \
                        if self.ma.compare_mode == GraphConst.MD5_COMPARE \
                        else max(precision_index, api.data.get(GraphConst.JSON_INDEX_KEY, GraphConst.MIN_INDEX_KEY))
                node.data[GraphConst.JSON_INDEX_KEY] = precision_index

    def _compare_nodes(self, node_n):
        """
        递归遍历NPU树中的节点，如果在Bench中找到具有相同名称的节点，检查他们的祖先和参数信息，检查一致则及逆行精度数据对比
        这里采用先序遍历，好处在于当这个节点被比较时，他的先序已经被匹配，这可以为后续的模糊匹配提供重要信息
        """
        if self.mapping_dict:
            node_b, ancestors_n, ancestors_b = Graph.mapping_match(node_n, self.graph_b, self.mapping_dict)
            if node_b:
                ancestors_n.append(node_n.id)
                ancestors_b.append(node_b.id)
                node_n.matched_node_link = ancestors_b
                node_b.matched_node_link = ancestors_n
        else:
            node_b, ancestors = Graph.match(self.graph_n, node_n, self.graph_b)
            if node_b:
                ancestors.append(node_b.id)
                node_n.add_link(node_b, ancestors)
        if node_b:
            # 真实数据比对只会得到基本信息，并没有精度指标，需要调用多进程对比接口
            self._get_and_add_result(node_n, node_b)
        for subnode in node_n.subnodes:
            self._compare_nodes(subnode)

    def _compare_nodes_fuzzy(self, node_n):
        if node_n.op != NodeOp.function_api:
            # 模块经过模糊匹配
            node_b, ancestors_n, ancestors_b = Graph.fuzzy_match(node_n, self.graph_b.node_map.get(node_n.id))
            if node_b:
                self._process_matched_nodes(node_n, node_b, ancestors_n, ancestors_b)
                # 匹配上的两个模块中的所有api, 忽略dump调用次数，按照名称一致+模块中的调用顺序进行匹配
                recount_result_n = self._recount_api_node(node_n)
                recount_result_b = self._recount_api_node(node_b)
                for recount_node_id, node_id_n in recount_result_n.items():
                    api_node_n = self.graph_n.node_map.get(node_id_n)
                    if not api_node_n:
                        continue
                    api_node_b, ancestors_n, ancestors_b = Graph.fuzzy_match(
                        api_node_n, self.graph_b.node_map.get(recount_result_b.get(recount_node_id)))
                    if api_node_b:
                        self._process_matched_nodes(api_node_n, api_node_b, ancestors_n, ancestors_b)
        for sub_node in node_n.subnodes:
            self._compare_nodes_fuzzy(sub_node)

    def _get_and_add_result(self, node_n, node_b):
        compare_result_list = compare_node([node_n.id, node_b.id],
                                           [self.data_n_dict, self.data_b_dict],
                                           self.stack_json_data, self.ma.compare_mode)
        if compare_result_list:
            self.ma.add_csv_data(compare_result_list)
            self.add_compare_result_to_node(node_n, compare_result_list)

    def _recount_api_node(self, node):
        """
        两个匹配上的模块, 忽略各自模块下所有api的dump调用次数, 并赋予模块中的调用顺序
        Return:
            {赋予模块中的调用顺序的node_id: 原始node_id}
        """
        recount_result = {}
        node_count = {}
        for sub_node in node.subnodes:
            if sub_node.op == NodeOp.function_api:
                # 忽略dump调用次数
                count_removed_id = self.pattern.sub(Const.SEP, sub_node.id)
                node_count[count_removed_id] = node_count.get(count_removed_id, 0) + 1
                # 赋予模块中的调用顺序
                recount_node_id = count_removed_id + str(node_count.get(count_removed_id))
                recount_result[recount_node_id] = sub_node.id
        return recount_result

    def _process_matched_nodes(self, node_n, node_b, ancestors_n, ancestors_b):
        ancestors_n.append(node_n.id)
        ancestors_b.append(node_b.id)
        node_n.matched_node_link = ancestors_b
        node_b.matched_node_link = ancestors_n
        self._get_and_add_result(node_n, node_b)
