# coding=utf-8
"""
# Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""

import unittest

from msprobe.core.common.const import CompareConst
from msprobe.core.compare.merge_result.utils import process_compare_index_dict_na, check_npu_bench_max_dtype


class TestProcessCompareIndexDictNa(unittest.TestCase):

    def setUp(self):
        # 初始化测试数据
        self.compare_index_dict = {
            'Max diff': {
                'op_name_1': {0: 'N/A'},
                'op_name_2': {0: 'N/A'}
            },
            'L2norm diff': {
                'op_name_1': {0: 'N/A'},
                'op_name_2': {0: 'N/A'}
            },
            'MeanRelativeErr': {
                'op_name_1': {0: 'N/A'},
                'op_name_2': {0: 'N/A'}
            },
            CompareConst.NPU_MAX: {
                'op_name_1': {0: 'tp-0-1-2-3'},
                'op_name_2': {0: 'tp-0-1-2-3'}
            },
            CompareConst.BENCH_MAX: {
                'op_name_1': {0: 'tp-0-1-2-3'},
                'op_name_2': {0: 'tp-0-1-2-3'}
            }
        }
        self.compare_index_list = ['Max diff', 'L2norm diff', 'MeanRelativeErr', 'NPU max', 'Bench max']
        self.rank_num = 0

    def test_process_compare_index_dict_na(self):
        result = process_compare_index_dict_na(self.compare_index_dict, self.compare_index_list, self.rank_num)

        # 检查是否替换了 N/A 值
        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], 'NPU:tp-0-1-2-3  Bench:tp-0-1-2-3')

    def test_no_na_values(self):
        # 修改测试数据，确保没有 N/A 值
        for index in self.compare_index_list[:-2]:  # 排除 'NPU max' 和 'Bench max'
            self.compare_index_dict[index] = {
                'op_name_1': {0: 'tp-0-1-2-3'},
                'op_name_2': {0: 'tp-0-1-2-3'}
            }

        result = process_compare_index_dict_na(self.compare_index_dict, self.compare_index_list, self.rank_num)

        # 验证返回值没有变化
        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], 'tp-0-1-2-3')
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], 'tp-0-1-2-3')

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], 'tp-0-1-2-3')
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], 'tp-0-1-2-3')

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], 'tp-0-1-2-3')
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], 'tp-0-1-2-3')

    def test_non_string_npu_bench(self):
        # 修改 NPU 和 Bench 统计量为非字符串类型
        self.compare_index_dict[CompareConst.NPU_MAX] = {
            'op_name_1': {0: 123},
            'op_name_2': {0: 123}
        }
        self.compare_index_dict[CompareConst.BENCH_MAX] = {
            'op_name_1': {0: 123},
            'op_name_2': {0: 123}
        }

        result = process_compare_index_dict_na(self.compare_index_dict, self.compare_index_list, self.rank_num)

        # 验证结果没有变化
        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], 'N/A')
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], 'N/A')

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], 'N/A')
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], 'N/A')

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], 'N/A')
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], 'N/A')

    def test_missing_npu_bench_max(self):
        # 移除 NPU_MAX 和 BENCH_MAX 键
        del self.compare_index_dict[CompareConst.NPU_MAX]
        del self.compare_index_dict[CompareConst.BENCH_MAX]

        result = process_compare_index_dict_na(self.compare_index_dict, self.compare_index_list, self.rank_num)

        # 验证原始数据未改变
        self.assertEqual(result['Max diff']['op_name_1'][self.rank_num], 'N/A')
        self.assertEqual(result['Max diff']['op_name_2'][self.rank_num], 'N/A')

        self.assertEqual(result['L2norm diff']['op_name_1'][self.rank_num], 'N/A')
        self.assertEqual(result['L2norm diff']['op_name_2'][self.rank_num], 'N/A')

        self.assertEqual(result['MeanRelativeErr']['op_name_1'][self.rank_num], 'N/A')
        self.assertEqual(result['MeanRelativeErr']['op_name_2'][self.rank_num], 'N/A')

    def test_same_type_and_valid(self):
        # 测试相同类型且在有效类型范围内
        self.assertTrue(check_npu_bench_max_dtype("test", "string"))
        self.assertTrue(check_npu_bench_max_dtype(True, False))
        self.assertTrue(check_npu_bench_max_dtype(None, None))

    def test_different_types(self):
        # 测试不同类型
        self.assertFalse(check_npu_bench_max_dtype("test", True))
        self.assertFalse(check_npu_bench_max_dtype("test", 123))
        self.assertFalse(check_npu_bench_max_dtype(None, "test"))

    def test_invalid_types(self):
        # 测试类型不属于有效类型
        self.assertFalse(check_npu_bench_max_dtype(123, 456))
        self.assertFalse(check_npu_bench_max_dtype(3.14, 2.71))

    def test_edge_cases(self):
        # 测试在有效类型范围内, 但类型不同
        self.assertFalse(check_npu_bench_max_dtype(None, "test"))
        self.assertFalse(check_npu_bench_max_dtype("test", None))
