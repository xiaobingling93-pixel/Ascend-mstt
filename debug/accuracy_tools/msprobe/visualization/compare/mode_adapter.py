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

import json
import math
from msprobe.core.common.const import CompareConst, Const
from msprobe.visualization.utils import ToolTip, GraphConst, str2float


class ModeAdapter:
    def __init__(self, compare_mode):
        self.compare_mode = compare_mode
        self.csv_data = []
        self.compare_nodes = []

    @staticmethod
    def _add_md5_compare_data(node_data, compare_data_dict):
        precision_index = GraphConst.MAX_INDEX_KEY
        for key, value in node_data.items():
            if not isinstance(value, dict):
                continue
            compare_data = compare_data_dict.get(key)
            if compare_data:
                headers = CompareConst.MD5_COMPARE_RESULT_HEADER
                id_list = [headers.index(x) for x in GraphConst.MD5_INDEX_LIST]
                ModeAdapter._match_data(value, compare_data, GraphConst.MD5_INDEX_LIST, id_list)
                # md5比对是否通过
                if value.get(CompareConst.RESULT) != CompareConst.PASS:
                    precision_index = GraphConst.MIN_INDEX_KEY
                node_data[key] = value
        return precision_index

    @staticmethod
    def _add_real_compare_data(node_data, compare_data_dict):
        min_thousandth = float(1)
        numbers = []
        for key, value in node_data.items():
            if not isinstance(value, dict):
                continue
            compare_data = compare_data_dict.get(key)
            if compare_data:
                headers = CompareConst.COMPARE_RESULT_HEADER
                id_list = [headers.index(x) for x in GraphConst.REAL_DATA_INDEX_LIST]
                ModeAdapter._match_data(value, compare_data, GraphConst.REAL_DATA_INDEX_LIST, id_list)
                # 跳过scalar data，因为无法计算双千指标，会得到Nan
                if not value.get(Const.SHAPE):
                    continue
                # 获取一个节点所有的输入或输出最小的双千指标
                thousandth = value.get(CompareConst.ONE_THOUSANDTH_ERR_RATIO)
                # 可能是None，可能是非数字内容str
                try:
                    thousandth = float(thousandth)
                except (ValueError, TypeError):
                    thousandth = None
                if thousandth is not None:
                    numbers.append(thousandth)
                node_data[key] = value
        # 双千指标都是None的异常情况
        if not numbers:
            min_thousandth = None
        else:
            min_thousandth = min(numbers + [min_thousandth])
        return min_thousandth

    @staticmethod
    def _add_summary_compare_data(node_data, compare_data_dict):
        max_relative_err = GraphConst.MIN_INDEX_KEY
        # data_info: {'type': 'torch.Tensor', 'dtype': 'torch.float32', 'shape': [2, 536320], 'Max': 9.66036224, ...}
        for key, data_info in node_data.items():
            if not isinstance(data_info, dict):
                continue
            compare_data = compare_data_dict.get(key)
            if compare_data:
                # 对应比对结果csv的列
                key_list = GraphConst.SUMMARY_INDEX_LIST
                headers = CompareConst.SUMMARY_COMPARE_RESULT_HEADER
                id_list = [headers.index(x) for x in key_list]
                ModeAdapter._match_data(data_info, compare_data, key_list, id_list)
                for item in key_list[4:]:
                    relative_err = str2float(data_info.get(item))
                    max_relative_err = max(max_relative_err, relative_err)
                node_data[key] = data_info
        max_relative_err = 1 if max_relative_err > 1 else max_relative_err
        return max_relative_err

    @staticmethod
    def _match_data(data_dict, compare_data, key_list, id_list):
        """
        绑定精度指标到node的input_data和output_data
        """
        if len(key_list) != len(id_list):
            return
        for id_val, key in zip(id_list, key_list):
            data_dict[key] = compare_data[id_val]

    @staticmethod
    def _check_list_len(data_list, len_num):
        if len(data_list) < len_num:
            raise ValueError(f"compare_data_dict_list must contain at least {len_num} items.")

    def parse_result(self, node, compare_data_dict_list):
        """
        根据结果返回数据，分别是precision_index，和附加数据
        """

        other_dict = {}
        if self.compare_mode == GraphConst.MD5_COMPARE:
            ModeAdapter._check_list_len(compare_data_dict_list, 2)
            precision_index_in = ModeAdapter._add_md5_compare_data(node.input_data, compare_data_dict_list[0])
            precision_index_out = ModeAdapter._add_md5_compare_data(node.output_data, compare_data_dict_list[1])
            # 所有输入输出md5对比通过，这个节点才算通过
            precision_index = min(precision_index_in, precision_index_out)
            other_result = CompareConst.PASS if precision_index == GraphConst.MAX_INDEX_KEY else CompareConst.DIFF
            other_dict[CompareConst.RESULT] = other_result
        elif self.compare_mode == GraphConst.SUMMARY_COMPARE:
            ModeAdapter._check_list_len(compare_data_dict_list, 2)
            ModeAdapter._add_summary_compare_data(node.input_data, compare_data_dict_list[0])
            precision_index_out = ModeAdapter._add_summary_compare_data(node.output_data, compare_data_dict_list[1])
            precision_index = precision_index_out
        else:
            ModeAdapter._check_list_len(compare_data_dict_list, 1)
            min_thousandth_in = ModeAdapter._add_real_compare_data(node.input_data, compare_data_dict_list[0])
            min_thousandth_out = ModeAdapter._add_real_compare_data(node.output_data, compare_data_dict_list[0])
            if min_thousandth_in is not None and min_thousandth_out is not None:
                change_percentage = min_thousandth_in - min_thousandth_out
            else:
                change_percentage = GraphConst.MIN_INDEX_KEY
            change_percentage = GraphConst.MIN_INDEX_KEY if change_percentage < GraphConst.MIN_INDEX_KEY \
                else change_percentage
            precision_index = GraphConst.MAX_INDEX_KEY \
                if change_percentage > GraphConst.MAX_INDEX_KEY else change_percentage
        return precision_index, other_dict

    def prepare_real_data(self, node):
        """
        为真实数据比较模式准备节点信息
        """
        if self.compare_mode == GraphConst.REAL_DATA_COMPARE:
            self.compare_nodes.append(node)
            return True
        return False

    def add_csv_data(self, compare_result_list):
        if self.compare_mode != GraphConst.REAL_DATA_COMPARE:
            return
        self.csv_data.extend(compare_result_list)

    def add_error_key(self, node_data):
        """
        根据不同的模式进行提供不同错误信息
        """
        for key, value in node_data.items():
            if not isinstance(value, dict):
                continue
            if self.compare_mode == GraphConst.SUMMARY_COMPARE:
                message = [CompareConst.MAX_RELATIVE_ERR, CompareConst.MIN_RELATIVE_ERR,
                           CompareConst.MEAN_RELATIVE_ERR, CompareConst.NORM_RELATIVE_ERR]
            elif self.compare_mode == GraphConst.REAL_DATA_COMPARE:
                message = [CompareConst.ONE_THOUSANDTH_ERR_RATIO, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO]
            else:
                # 输出件优化
                message = []
            value[GraphConst.ERROR_KEY] = message
            node_data[key] = value

    def get_tool_tip(self):
        """
        用于前端展示字段的具体含义
        """
        if self.compare_mode == GraphConst.SUMMARY_COMPARE:
            tips = {
                CompareConst.MAX_DIFF: ToolTip.MAX_DIFF,
                CompareConst.MIN_DIFF: ToolTip.MIN_DIFF,
                CompareConst.MEAN_DIFF: ToolTip.MEAN_DIFF,
                CompareConst.NORM_DIFF: ToolTip.NORM_DIFF}
        elif self.compare_mode == GraphConst.MD5_COMPARE:
            tips = {Const.MD5: ToolTip.MD5}
        else:
            tips = {
                CompareConst.ONE_THOUSANDTH_ERR_RATIO: ToolTip.ONE_THOUSANDTH_ERR_RATIO,
                CompareConst.FIVE_THOUSANDTHS_ERR_RATIO: ToolTip.FIVE_THOUSANDTHS_ERR_RATIO,
                CompareConst.COSINE: ToolTip.COSINE,
                CompareConst.MAX_ABS_ERR: ToolTip.MAX_ABS_ERR,
                CompareConst.MAX_RELATIVE_ERR: ToolTip.MAX_RELATIVE_ERR}
        return json.dumps(tips)
