# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

from msprobe.core.common.const import CompareConst


def replace_compare_index_dict(compare_index_dict, compare_index_list, rank_num):
    """
    比对指标值为N/A、unsupported、Nan，将比对指标值替换成NPU max 和 Bench max(几个统计量相同)

    示例：
    Distributed.all_reduce.0.forward.output.group的比对指标值是N/A
    替换后：
    比对指标值为:
        NPU: tp-0-1-2-3
        Bench: tp-0-1-2-3
    """

    if CompareConst.NPU_MAX not in compare_index_dict or CompareConst.BENCH_MAX not in compare_index_dict:
        compare_index_dict.pop(CompareConst.NPU_MAX, None)
        compare_index_dict.pop(CompareConst.BENCH_MAX, None)
        return compare_index_dict

    # 遍历比对指标列表，排除最后两个指标NPU max， Bench max
    for compare_index in compare_index_list[:-2]:
        op_name_index_dict = compare_index_dict[compare_index]
        # 遍历op_item名称和对应的比对指标值
        for op_name, index_value in op_name_index_dict.items():
            npu_max = compare_index_dict[CompareConst.NPU_MAX][op_name][rank_num]
            bench_max = compare_index_dict[CompareConst.BENCH_MAX][op_name][rank_num]
            # 如果当前比对指标值是N/A、unsupported、Nan，并且NPU和Bench的最大值是类型相同，进行替换
            if index_value[rank_num] in [CompareConst.N_A, CompareConst.UNSUPPORTED, CompareConst.NAN]:
                compare_index_dict[compare_index][op_name][rank_num] = f'NPU:{str(npu_max)}  Bench:{str(bench_max)}'

    # 删除NPU_MAX和BENCH_MAX
    compare_index_dict.pop(CompareConst.NPU_MAX, None)
    compare_index_dict.pop(CompareConst.BENCH_MAX, None)
    return compare_index_dict
