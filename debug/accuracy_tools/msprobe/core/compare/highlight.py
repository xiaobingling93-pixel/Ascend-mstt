# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

import math
import abc
import re
from collections import namedtuple
import numpy as np
import openpyxl
from openpyxl.styles import PatternFill
from msprobe.core.common.utils import get_header_index
from msprobe.core.common.file_utils import save_workbook
from msprobe.core.common.log import logger
from msprobe.core.common.const import CompareConst, FileCheckConst


class HighlightCheck(abc.ABC):
    @abc.abstractmethod
    def apply(self, info, color_columns, summary_compare):
        raise NotImplementedError


class CheckOrderMagnitude(HighlightCheck):
    """检查Max diff的数量级差异"""
    def apply(self, info, color_columns, summary_compare=True):
        api_in, api_out, num = info
        max_diff_index = get_header_index('Max diff' if summary_compare else 'MaxAbsErr', summary_compare)
        if abs(api_in[max_diff_index]) > abs(api_out[max_diff_index]):
            return
        in_order = 0 if abs(api_in[max_diff_index]) < 1 else math.log10(abs(api_in[max_diff_index]))
        out_order = 0 if abs(api_out[max_diff_index]) < 1 else math.log10(abs(api_out[max_diff_index]))
        if out_order - in_order >= CompareConst.ORDER_MAGNITUDE_DIFF_YELLOW:
            color_columns.yellow.append(num)


class CheckOneThousandErrorRatio(HighlightCheck):
    """检查千分误差比率"""
    def apply(self, info, color_columns, summary_compare=True):
        api_in, api_out, num = info
        one_thousand_index = get_header_index('One Thousandth Err Ratio', summary_compare)
        if (not isinstance(api_in[one_thousand_index], (float, int)) or
                not isinstance(api_out[one_thousand_index], (float, int))):
            return
        if (api_in[one_thousand_index] > CompareConst.ONE_THOUSAND_ERROR_IN_RED and
                api_out[one_thousand_index] < CompareConst.ONE_THOUSAND_ERROR_OUT_RED):
            color_columns.red.append(num)
        elif api_in[one_thousand_index] - api_out[one_thousand_index] > CompareConst.ONE_THOUSAND_ERROR_DIFF_YELLOW:
            color_columns.yellow.append(num)


class CheckCosineSimilarity(HighlightCheck):
    """检查余弦相似度"""
    def apply(self, info, color_columns, summary_compare=True):
        api_in, api_out, num = info
        cosine_index = get_header_index('Cosine', summary_compare)
        if not isinstance(api_in[cosine_index], (float, int)) or not isinstance(api_out[cosine_index], (float, int)):
            return
        if api_in[cosine_index] - api_out[cosine_index] > CompareConst.COSINE_DIFF_YELLOW:
            color_columns.yellow.append(num)


class CheckMaxRelativeDiff(HighlightCheck):
    """检查最大相对差异"""
    def apply(self, info, color_columns, summary_compare=True):
        api_in, api_out, num = info
        max_diff_index = get_header_index('Max diff', summary_compare)
        bench_max_index = get_header_index('Bench max', summary_compare)
        input_max_relative_diff = np.abs(np.divide(api_in[max_diff_index], max(0.01, api_in[bench_max_index])))
        output_max_relative_diff = np.abs(np.divide(api_out[max_diff_index], max(0.01, api_out[bench_max_index])))
        if not isinstance(input_max_relative_diff, (float, int)) or not isinstance(output_max_relative_diff,
                                                                                   (float, int)):
            return
        if output_max_relative_diff > CompareConst.MAX_RELATIVE_OUT_RED:
            color_columns.red.append(num)
        elif (output_max_relative_diff > CompareConst.MAX_RELATIVE_OUT_YELLOW and
              input_max_relative_diff < CompareConst.MAX_RELATIVE_IN_YELLOW):
            color_columns.yellow.append(num)


class CheckOverflow(HighlightCheck):
    """检查是否存在溢出"""
    def apply(self, info, color_columns, summary_compare=True):
        line, num = info
        npu_max_index = get_header_index('NPU max', summary_compare)
        npu_min_index = get_header_index('NPU min', summary_compare)
        max_diff_index = get_header_index('Max diff' if summary_compare else 'MaxAbsErr', summary_compare)
        if str(line[npu_max_index]) in CompareConst.OVERFLOW_LIST or str(
                line[npu_min_index]) in CompareConst.OVERFLOW_LIST:
            color_columns.red.append(num)
            return
        # check if Max_Diff > 1e+10
        if isinstance(line[max_diff_index], (float, int)) and line[max_diff_index] > CompareConst.MAX_DIFF_RED:
            color_columns.red.append(num)


class HighlightRules:
    """高亮规则集合，用于检查API的误差"""
    # 适用于每行的规则
    basic_rules = {
        "check_overflow": CheckOverflow()
    }

    # 用于比较输入和输出的规则
    compare_rules = {
        "check_order_magnitude": CheckOrderMagnitude(),
        "check_one_thousand_error": CheckOneThousandErrorRatio(),
        "check_cosine_similarity": CheckCosineSimilarity()
    }
    summary_compare_rules = {
        "check_order_magnitude": CheckOrderMagnitude(),
        "check_max_relative_diff": CheckMaxRelativeDiff(),
    }
    

def find_error_rows(result, last_len, n_num_input, highlight_dict, summary_compare=False, md5_compare=False):
    """找到单个API中需要高亮的行"""
    if md5_compare:
        return
    npu_max_index = get_header_index('NPU max', summary_compare)
    bench_max_index = get_header_index('Bench max', summary_compare)
    max_diff_index = get_header_index('Max diff' if summary_compare else 'MaxAbsErr', summary_compare)

    red_lines, yellow_lines = [], []
    LineInfo = namedtuple('LineInfo', ['line_data', 'num_pointer'])
    ApiInfo = namedtuple('ApiInfo', ['api_input', 'api_output', 'num_pointer'])
    ColorColumns = namedtuple('ColorColumns', ['red', 'yellow'])
    color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)

    # 对单行API的输入或输出进行误差判断
    for i, line in enumerate(result):
        num = last_len + i
        line_info = LineInfo(line_data=line, num_pointer=num)
        for rule in HighlightRules.basic_rules.values():
            rule.apply(line_info, color_columns, summary_compare)

    # 对API的输出与输入比较，进行误差判断
    for n, api_out in enumerate(result[n_num_input:len(result)]):
        num = last_len + n_num_input + n
        if num in red_lines:
            continue
        if not isinstance(api_out[npu_max_index], (float, int)) \
                or not isinstance(api_out[bench_max_index], (float, int)) \
                or not isinstance(api_out[max_diff_index], (float, int)):
            continue
        for _, api_in in enumerate(result[0:n_num_input]):
            if not isinstance(api_in[npu_max_index], (float, int)) \
                    or not isinstance(api_in[bench_max_index], (float, int)) \
                    or not isinstance(api_in[max_diff_index], (float, int)):
                continue

            api_info = ApiInfo(api_input=api_in, api_output=api_out, num_pointer=num)
            if summary_compare:
                for rule in HighlightRules.summary_compare_rules.values():
                    rule.apply(api_info, color_columns, summary_compare)
            else:
                for rule in HighlightRules.compare_rules.values():
                    rule.apply(api_info, color_columns, summary_compare)

    highlight_dict.get('red_rows', []).extend(list(set(red_lines)))
    highlight_dict.get('yellow_rows', []).extend(list(set(yellow_lines) - set(red_lines)))


def get_name_and_state(name):
    """Get api/module name and state"""
    if "input" in name:
        api_name = name.split("input")[0]
        state = "input"
    else:
        api_name = name.split("output")[0]
        state = "output"
    return api_name, state


def find_compare_result_error_rows(result_df, highlight_dict, summary_compare, md5_compare):
    """将dataframe根据API分组，并找到有误差的算子用于高亮"""
    result = result_df.values
    start, input_num, output_num, end = 0, 0, 0, len(result_df)
    last_api_name, last_state = None, None
    num, last_len = 0, 0
    for res_i in result:
        api_name, state = get_name_and_state(res_i[0])
        if last_api_name:
            if api_name == last_api_name:
                if state == last_state:
                    num += 1
                else:
                    input_num = num
                    num, last_state = 1, state
            else:
                output_num = num
                find_error_rows(result[start:start + input_num + output_num], start, input_num, highlight_dict,
                                summary_compare, md5_compare)
                num, last_api_name, last_state = 1, api_name, state
                start += input_num + output_num
                input_num, output_num = 1, 0
        else:
            num, last_api_name, last_state = 1, api_name, state
    if state:
        if state == "input":
            input_num = num
        else:
            output_num = num
        find_error_rows(result[start:start + input_num + output_num], start, input_num, highlight_dict,
                        summary_compare, md5_compare)


def highlight_rows_xlsx(result_df, highlight_dict, file_path):
    """Write and highlight results in Excel"""
    logger.info('Compare result is %s' % file_path)

    wb = openpyxl.Workbook()
    ws = wb.active

    # write header
    for j, col_name in enumerate(result_df.columns, start=1):
        if not csv_value_is_valid(col_name):
            raise RuntimeError(f"Malicious value [{col_name}] is not allowed to be written into the xlsx: {file_path}.")
        ws.cell(row=1, column=j, value=col_name)

    for i, row in enumerate(result_df.iterrows(), start=2):
        for j, value in enumerate(row[1], start=1):
            if not isinstance(value, (float, int)):
                value = f'{str(value)}\t' if str(value) in ('inf', '-inf', 'nan') else str(value)
            if not csv_value_is_valid(value):
                raise RuntimeError(f"Malicious value [{value}] is not allowed to be written into the xlsx: {file_path}.")
            ws.cell(row=i, column=j, value=f'{str(value)}\t' if str(value) in ('inf', '-inf', 'nan') else value)

            if (i - 2) in highlight_dict['red_rows']:
                ws.cell(row=i, column=j).fill = PatternFill(start_color=CompareConst.RED,
                                                            end_color=CompareConst.RED, fill_type="solid")
            elif (i - 2) in highlight_dict['yellow_rows']:
                ws.cell(row=i, column=j).fill = PatternFill(start_color=CompareConst.YELLOW,
                                                            end_color=CompareConst.YELLOW, fill_type="solid")

    save_workbook(wb, file_path)


def csv_value_is_valid(value: str) -> bool:
    if not isinstance(value, str):
        return True
    try:
        # -1.00 or +1.00 should be consdiered as digit numbers
        float(value)
    except ValueError:
        # otherwise, they will be considered as formular injections
        return not bool(re.compile(FileCheckConst.CSV_BLACK_LIST).search(value))
    return True
