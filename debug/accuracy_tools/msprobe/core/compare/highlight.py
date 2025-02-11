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

import abc
import math
import multiprocessing
import re
from collections import namedtuple

import numpy as np
import openpyxl
from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from tqdm import tqdm

from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.file_utils import save_workbook
from msprobe.core.common.log import logger
from msprobe.core.common.utils import get_header_index, safe_get_value
from msprobe.core.compare.utils import table_value_is_valid, get_name_and_state, CompareException


class HighlightCheck(abc.ABC):
    @abc.abstractmethod
    def apply(self, info, color_columns, dump_mode):
        raise NotImplementedError


def add_highlight_row_info(color_list, num, highlight_err_msg):
    for i, (existing_num, existing_err_msg) in enumerate(color_list):
        if num == existing_num:
            color_list[i][1].append(highlight_err_msg)
            return
    color_list.append((num, [highlight_err_msg]))


class CheckOrderMagnitude(HighlightCheck):
    """检查Max diff的数量级差异"""

    def apply(self, info, color_columns, dump_mode):
        api_in, api_out, num = info
        max_diff_index = get_header_index(CompareConst.MAX_DIFF if dump_mode == Const.SUMMARY
                                          else CompareConst.MAX_ABS_ERR, dump_mode)
        if abs(api_in[max_diff_index]) > abs(api_out[max_diff_index]):
            return
        in_order = 0 if abs(api_in[max_diff_index]) < 1 else math.log10(abs(api_in[max_diff_index]))
        out_order = 0 if abs(api_out[max_diff_index]) < 1 else math.log10(abs(api_out[max_diff_index]))
        if out_order - in_order >= CompareConst.ORDER_MAGNITUDE_DIFF_YELLOW:
            add_highlight_row_info(color_columns.yellow, num,
                                   "maximum absolute error of both input/parameters and output exceed 1, "
                                   "with the output larger by an order of magnitude")


class CheckOneThousandErrorRatio(HighlightCheck):
    """检查千分误差比率"""

    def apply(self, info, color_columns, dump_mode):
        api_in, api_out, num = info
        one_thousand_index = get_header_index(CompareConst.ONE_THOUSANDTH_ERR_RATIO, dump_mode)
        if (not isinstance(api_in[one_thousand_index], (float, int)) or
                not isinstance(api_out[one_thousand_index], (float, int))):
            return
        if (api_in[one_thousand_index] > CompareConst.ONE_THOUSAND_ERROR_IN_RED and
                api_out[one_thousand_index] < CompareConst.ONE_THOUSAND_ERROR_OUT_RED):
            add_highlight_row_info(color_columns.red, num,
                                   "The input/parameters's one thousandth err ratio exceeds 0.9, "
                                   "while the output's is below 0.6")
        elif api_in[one_thousand_index] - api_out[one_thousand_index] > CompareConst.ONE_THOUSAND_ERROR_DIFF_YELLOW:
            add_highlight_row_info(color_columns.yellow, num,
                                   "The output's one thousandth err ratio decreases by more than 0.1 "
                                   "compared to the input/parameters's")


class CheckCosineSimilarity(HighlightCheck):
    """检查余弦相似度"""

    def apply(self, info, color_columns, dump_mode):
        api_in, api_out, num = info
        cosine_index = get_header_index(CompareConst.COSINE, dump_mode)
        if not isinstance(api_in[cosine_index], (float, int)) or not isinstance(api_out[cosine_index], (float, int)):
            return
        if api_in[cosine_index] - api_out[cosine_index] > CompareConst.COSINE_DIFF_YELLOW:
            add_highlight_row_info(color_columns.yellow, num,
                                   "The output's cosine decreases by more than 0.1 "
                                   "compared to the input/parameters's")


class CheckMaxRelativeDiff(HighlightCheck):
    """检查最大相对差异"""

    def apply(self, info, color_columns, dump_mode):
        api_in, api_out, num = info
        max_diff_index = get_header_index(CompareConst.MAX_DIFF, dump_mode)
        bench_max_index = get_header_index(CompareConst.BENCH_MAX, dump_mode)
        input_max_relative_diff = np.abs(
            np.divide(api_in[max_diff_index], max(Const.FLOAT_EPSILON, api_in[bench_max_index])))
        output_max_relative_diff = np.abs(
            np.divide(api_out[max_diff_index], max(Const.FLOAT_EPSILON, api_out[bench_max_index])))
        if not isinstance(input_max_relative_diff, (float, int)) or not isinstance(output_max_relative_diff,
                                                                                   (float, int)):
            return
        if output_max_relative_diff > CompareConst.MAX_RELATIVE_OUT_RED:
            add_highlight_row_info(color_columns.red, num, "maximum relative error exceeds 0.5")
        elif (output_max_relative_diff > CompareConst.MAX_RELATIVE_OUT_YELLOW and
              input_max_relative_diff < CompareConst.MAX_RELATIVE_IN_YELLOW):
            add_highlight_row_info(color_columns.yellow, num,
                                   "The output's maximum relative error exceeds 0.1, "
                                   "while the input/parameters's is below 0.01")


class CheckOverflow(HighlightCheck):
    """检查是否存在溢出"""

    def apply(self, info, color_columns, dump_mode):
        line, num = info
        npu_max_index = get_header_index(CompareConst.NPU_MAX, dump_mode)
        npu_min_index = get_header_index(CompareConst.NPU_MIN, dump_mode)
        max_diff_index = get_header_index(CompareConst.MAX_DIFF if dump_mode == Const.SUMMARY
                                          else CompareConst.MAX_ABS_ERR, dump_mode)
        if str(line[npu_max_index]) in CompareConst.OVERFLOW_LIST or str(
                line[npu_min_index]) in CompareConst.OVERFLOW_LIST:
            add_highlight_row_info(color_columns.red, num, "maximum or minimum is nan, -inf, or inf")
            return
        # check if Max_Diff > 1e+10
        if isinstance(line[max_diff_index], (float, int)) and abs(line[max_diff_index]) > CompareConst.MAX_DIFF_RED:
            add_highlight_row_info(color_columns.red, num, "maximum absolute error exceeds 1e+10")


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


def check_indices_numeric(api_items, indices: list):
    """检查指定索引处的值是否都为数字类型（int 或 float）"""
    return all(isinstance(api_items[i], (float, int)) for i in indices)


def apply_comparison_rules(api_info, dump_mode, color_columns):
    """output与input/params的比较"""
    if dump_mode == Const.SUMMARY:
        for rule in HighlightRules.summary_compare_rules.values():
            rule.apply(api_info, color_columns, dump_mode)
    else:
        for rule in HighlightRules.compare_rules.values():
            rule.apply(api_info, color_columns, dump_mode)


def find_error_rows(result, api_batch, highlight_dict, dump_mode):
    """找到单个API中需要高亮的行"""
    if dump_mode == Const.MD5:
        return
    npu_max_index = get_header_index(CompareConst.NPU_MAX, dump_mode)
    bench_max_index = get_header_index(CompareConst.BENCH_MAX, dump_mode)
    max_diff_index = get_header_index(CompareConst.MAX_DIFF if dump_mode == Const.SUMMARY
                                      else CompareConst.MAX_ABS_ERR, dump_mode)

    red_lines, yellow_lines = [], []
    LineInfo = namedtuple('LineInfo', ['line_data', 'num_pointer'])
    ApiInfo = namedtuple('ApiInfo', ['api_input', 'api_output', 'num_pointer'])
    ColorColumns = namedtuple('ColorColumns', ['red', 'yellow'])
    color_columns = ColorColumns(red=red_lines, yellow=yellow_lines)

    api_batch_start = api_batch.start  # result_df的input起始全局索引
    api_batch_params_end_index = api_batch.params_end_index  # result_df的params结束全局索引 + 1
    api_batch_output_end_index = api_batch.output_end_index  # result_df的output结束全局索引 + 1
    api_batch_params_slice_index_local = api_batch_params_end_index - api_batch_start  # result的params结束局部切片索引
    api_batch_output_slice_index_local = api_batch_output_end_index - api_batch_start  # result的output结束局部切片索引

    # 对单行API的输入或输出进行误差判断
    for i, line in enumerate(result):
        index = api_batch_start + i
        line_info = LineInfo(line_data=line, num_pointer=index)
        for rule in HighlightRules.basic_rules.values():
            rule.apply(line_info, color_columns, dump_mode)

    # 对API的输出与输入比较，进行误差判断
    for n, api_out in enumerate(result[api_batch_params_slice_index_local: api_batch_output_slice_index_local]):
        index = api_batch_start + api_batch_params_slice_index_local + n
        # 单行检查只有溢出检查（红色），如果已经溢出，不进一步检查
        if index in red_lines:
            continue
        if not check_indices_numeric(api_out, [npu_max_index, bench_max_index, max_diff_index]):
            continue

        # input/parameters的比较检查, 这里api_in包括input、parameters
        for _, api_in in enumerate(result[0: api_batch_params_slice_index_local]):
            if not check_indices_numeric(api_in, [npu_max_index, bench_max_index, max_diff_index]):
                continue
            api_info = ApiInfo(api_input=api_in, api_output=api_out, num_pointer=index)
            apply_comparison_rules(api_info, dump_mode, color_columns)

    red_lines_num_set = {x[0] for x in red_lines}
    yellow_lines_num_set = {x[0] for x in yellow_lines}
    highlight_dict.get('red_rows', set()).update(red_lines_num_set)
    highlight_dict.get('yellow_rows', set()).update(yellow_lines_num_set - red_lines_num_set)
    highlight_dict.get('red_lines', []).extend(red_lines)
    highlight_dict.get('yellow_lines', []).extend(yellow_lines)


class ApiBatch:
    def __init__(self, api_name: str, start: int):
        self.api_name = api_name
        self.start = start
        self.input_len = 1  # input的数量
        self.params_end_index = start + 1  # params的结束index
        self.output_end_index = start + 1  # output的结束index
        self.params_grad_end_index = start + 1  # params_grad的结束index
        # 内部state的标志("input", "output", "parameters", "parameters_grad"),
        # 用于控制计算input_len, output_end_index, params_end_index, self.params_grad_end_index
        self._state = Const.INPUT  # api_batch初始化为input

    def set_state(self, state: str):
        """设置当前状态"""
        if state in {Const.INPUT, Const.OUTPUT, Const.KWARGS, Const.PARAMS, Const.PARAMS_GRAD}:
            self._state = state
        else:
            raise ValueError(f"Invalid state: {state}")

    def increment(self, state: str):
        self.set_state(state)
        if self._state == Const.INPUT or self._state == Const.KWARGS:
            self.input_len += 1
            self.params_end_index += 1
            self.output_end_index += 1
        if self._state == Const.PARAMS:
            self.params_end_index += 1
            self.output_end_index += 1
        if self._state == Const.OUTPUT:
            self.output_end_index += 1
        self.params_grad_end_index += 1


def api_batches_update(api_batches, api_name, state, index):
    """
    当一个api的所有item更新完后，input, output的索引范围：
    input: [start: start+input_len]
    output: [start+input_len: output_end_index]
    params: [output_end_index: params_end_index]
    """
    if not api_batches:
        api_batches.append(ApiBatch(api_name, index))
    else:
        api_batch = api_batches[-1]
        if api_batch.api_name == api_name or (
                not re.search(Const.REGEX_FORWARD_BACKWARD, api_name) and api_name in api_batch.api_name):
            try:
                api_batch.increment(state)
            except ValueError as e:
                logger.error(f"api_batch: {api_batch} with invalid state, please check! {e}")
                raise CompareException(CompareException.INVALID_STATE_ERROR) from e
        else:
            api_batches.append(ApiBatch(api_name, index))


def find_compare_result_error_rows(result_df, highlight_dict, dump_mode):
    """将dataframe根据API分组，并找到有误差的算子用于高亮"""
    result = result_df.values
    api_batches = []
    for i, res_i in enumerate(result):
        api_full_name = safe_get_value(res_i, 0, "res_i")
        api_name, state = get_name_and_state(api_full_name)
        api_batches_update(api_batches, api_name, state, i)
    with tqdm(total=len(api_batches), desc="API/Module Analyse Progress", unit="item", ncols=100) as progress_bar:
        for api_batch in api_batches:
            find_error_rows(result[api_batch.start: api_batch.params_grad_end_index], api_batch, highlight_dict, 
                            dump_mode)
            progress_bar.update(1)


def value_check(value, api_name=None, i=None, result_df_columns=None):
    if not table_value_is_valid(value):
        if result_df_columns:
            logger.error(f"Malicious value [{value}] at api_name [{api_name}], column [{result_df_columns[i]}], "
                         f"is not allowed to be written into the compare result xlsx.")
        else:
            logger.error(f"Malicious value [{value}] is not allowed to be written into the compare result xlsx.")


def df_malicious_value_check(df_chunk, result_df_columns):
    for row in df_chunk.itertuples(index=False):
        api_name = row[0]
        for i, value in enumerate(row):
            value_check(value, api_name, i, result_df_columns)


def handle_multi_process_malicious_value_check(func, result_df):
    result_total_nums = len(result_df)
    process_num = int((multiprocessing.cpu_count() + 1) / 2)

    if result_total_nums <= process_num:
        process_num = 1
        chunks = [result_df]
    else:
        chunk_size = result_total_nums // process_num
        chunks = [result_df.iloc[i: i + chunk_size] for i in range(0, result_total_nums, chunk_size)]

    pool = multiprocessing.Pool(process_num)

    def err_call(args):
        logger.error("Multiprocessing malicious value check failed! Reason: {}".format(args))
        try:
            pool.terminate()
        except OSError:
            logger.error("Pool terminate failed")

    result_df_columns = result_df.columns.tolist()
    for column in result_df_columns:
        value_check(column)
    for df_chunk in chunks:
        pool.apply_async(func, args=(df_chunk, result_df_columns,), error_callback=err_call)

    pool.close()
    pool.join()


def compare_result_df_convert(value):
    if not isinstance(value, (float, int)) or isinstance(value, bool):  # bool类型或者非数字类型转str
        value = f"{str(value)}\t" if str(value) in ("inf", "-inf", "nan") else str(value)
    if isinstance(value, float):
        value = f"{str(value)}\t" if str(value) in ("inf", "-inf", "nan") else value
    return value


def highlight_rows_xlsx(result_df, highlight_dict, file_path):
    """Write and highlight results in Excel"""

    update_highlight_err_msg(result_df, highlight_dict)  # add highlight err_msg

    wb = openpyxl.Workbook()
    ws = wb.active

    # write header
    logger.info('Initializing Excel file.')

    handle_multi_process_malicious_value_check(df_malicious_value_check, result_df)

    result_df_convert = result_df.applymap(compare_result_df_convert)

    for row in dataframe_to_rows(result_df_convert, index=False, header=True):
        ws.append(row)

    # 对可疑数据标色
    logger.info('Coloring Excel in progress.')
    col_len = len(result_df.columns)
    red_fill = PatternFill(
        start_color=CompareConst.RED, end_color=CompareConst.RED, fill_type="solid"
    )
    yellow_fill = PatternFill(
        start_color=CompareConst.YELLOW, end_color=CompareConst.YELLOW, fill_type="solid",
    )
    for i in highlight_dict.get("red_rows", []):
        for j in range(1, col_len + 1):
            ws.cell(row=i + 2, column=j).fill = red_fill  # 2因为ws.cell中的row或column需要>=1,数据从第2行开始
    for i in highlight_dict.get("yellow_rows", []):
        for j in range(1, col_len + 1):
            ws.cell(row=i + 2, column=j).fill = yellow_fill

    logger.info('Saving Excel file to disk: %s' % file_path)
    save_workbook(wb, file_path)


def update_highlight_err_msg(result_df, highlight_dict):
    if result_df.shape[1] <= 1:
        return

    if CompareConst.NPU_MD5 in result_df.columns:
        return

    err_msg = result_df.get(CompareConst.ERROR_MESSAGE)
    red_lines_num_set = highlight_dict.get('red_rows')

    for color in ['red', 'yellow']:
        line_key = f'{color}_lines'
        lines = highlight_dict.get(line_key, [])
        for line_index, messages in lines:
            if color == 'yellow' and line_index in red_lines_num_set:
                continue  # 如果是 yellow 行，且已被 red 行覆盖，跳过

            for msg in messages:
                if err_msg[line_index] == '':
                    err_msg[line_index] = msg
                else:
                    err_msg[line_index] += '\n' + msg

            if color == 'red':
                red_lines_num_set.add(line_index)

    result_df[CompareConst.ERROR_MESSAGE] = err_msg
