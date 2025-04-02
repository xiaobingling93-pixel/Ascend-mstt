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

import multiprocessing
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from msprobe.core.advisor.advisor import Advisor
from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import load_json, remove_path, create_directory
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException, add_time_with_xlsx, check_op_str_pattern_valid, safe_get_value, \
    set_dump_path, get_dump_mode, check_compare_param, check_configuration_param
from msprobe.core.compare.check import check_dump_json_str, check_graph_mode, check_stack_json_str, \
    check_struct_match, fuzzy_check_op, cross_dtype_mapping_2
from msprobe.core.compare.highlight import find_compare_result_error_rows, highlight_rows_xlsx
from msprobe.core.compare.multiprocessing_compute import ComparisonResult, _handle_multi_process, _save_cmp_result
from msprobe.core.compare.npy_compare import compare_ops_apply, get_error_flag_and_msg
from msprobe.core.compare.utils import get_accuracy, get_rela_diff_summary_mode, get_un_match_accuracy, merge_tensor, \
    print_compare_ends_info, read_op, get_name_and_state, reorder_op_x_list, set_stack_json_path
from msprobe.core.compare.config import MappingConfig, MappingDict


@dataclass
class ComparisonConfig:
    dump_mode: str
    stack_mode: bool
    auto_analyze: bool
    fuzzy_match: bool
    data_mapping: dict
    suffix: str
    cell_mapping: dict
    api_mapping: dict
    layer_mapping: dict


class ModeConfig:
    def __init__(self, stack_mode=False, auto_analyze=True, fuzzy_match=False, dump_mode=None):
        self.stack_mode = stack_mode
        self.auto_analyze = auto_analyze
        self.fuzzy_match = fuzzy_match
        self.dump_mode = dump_mode


class Comparator:
    def __init__(self, mode_config: ModeConfig):
        self.stack_mode = mode_config.stack_mode
        self.auto_analyze = mode_config.auto_analyze
        self.fuzzy_match = mode_config.fuzzy_match
        self.dump_mode = mode_config.dump_mode

    @staticmethod
    def get_result_md5_compare(ms_op_name, bench_op_name, npu_ops_all, bench_ops_all, *args):
        npu_struct = npu_ops_all.get(ms_op_name).get('struct', [])
        bench_struct = bench_ops_all.get(bench_op_name).get('struct', [])

        if len(npu_struct) < 3 or len(bench_struct) < 3:
            logger.error(f"The length of npu_struct and bench_struct must be >= 3, "
                         f"but got npu_struct={len(npu_struct)} and bench_struct={len(bench_struct)}. Please check!")
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR)

        result_item = [ms_op_name, bench_op_name, npu_struct[0], bench_struct[0],
                       npu_struct[1], bench_struct[1], npu_struct[2], bench_struct[2],
                       CompareConst.PASS if npu_struct[2] == bench_struct[2] else CompareConst.DIFF]

        if len(args) >= 2 and args[0]:
            result_item.extend(args[1])
        else:
            result_item.append(CompareConst.NONE)
        return result_item

    @staticmethod
    def calculate_summary_data(npu_summary_data, bench_summary_data, result_item):
        err_msg = ""
        result_item, accuracy_check, err_msg = get_rela_diff_summary_mode(result_item, npu_summary_data,
                                                                          bench_summary_data, err_msg)
        result_item.append(accuracy_check)
        result_item.append(err_msg)

    @staticmethod
    def _generate_na_data(ops_all):
        if not ops_all:
            return {}
        key = next(iter(ops_all))
        value = deepcopy(ops_all[key])
        for k, v in value.items():
            if isinstance(v, tuple):
                value[k] = tuple(CompareConst.N_A for _ in range(len(v)))
            elif isinstance(v, list):
                value[k] = [CompareConst.N_A] * len(v)
            else:
                value[k] = CompareConst.N_A
        return value

    def make_result_table(self, result):
        header = CompareConst.HEAD_OF_COMPARE_MODE[self.dump_mode][:]

        if self.stack_mode:
            header.append(CompareConst.STACK)
            if self.dump_mode == Const.ALL:
                header.append(CompareConst.DATA_NAME)
        else:
            if self.dump_mode == Const.ALL:
                for row in result:
                    del row[-2]  # 输出结果不要堆栈信息时，删除中间结果result中的stack info，真实数据时为倒数第2列
                header.append(CompareConst.DATA_NAME)
            else:
                for row in result:
                    del row[-1]  # 输出结果不要堆栈信息时，删除中间结果result中的stack info，非真实数据时为倒数第1列
        result_df = pd.DataFrame(result, columns=header, dtype='object')
        return result_df

    def gen_merge_list(self, json_data, op_name, stack_json_data):
        op_data = json_data['data'][op_name]
        check_dump_json_str(op_data, op_name)
        op_parsed_list = read_op(op_data, op_name)

        if self.stack_mode:
            stack_info = stack_json_data.get(op_name)
            if stack_info is not None:
                check_stack_json_str(stack_info, op_name)
            # append only when stack_mode is True,
            op_parsed_list.append({
                'full_op_name': op_name,
                'full_info': stack_info
            })

        merge_list = merge_tensor(op_parsed_list, self.dump_mode)
        return merge_list

    def check_op(self, npu_dict, bench_dict):
        npu_op_name = npu_dict[CompareConst.OP_NAME]
        bench_op_name = bench_dict[CompareConst.OP_NAME]
        graph_mode = check_graph_mode(safe_get_value(npu_op_name, 0, "npu_op_name"),
                                      safe_get_value(bench_op_name, 0, "bench_op_name"))

        frame_name = getattr(self, "frame_name")
        if frame_name == "PTComparator":
            from msprobe.pytorch.compare.match import graph_mapping
            if graph_mode:
                return graph_mapping.match(npu_op_name[0], bench_op_name[0])
        struct_match = check_struct_match(npu_dict, bench_dict)
        if not self.fuzzy_match:
            name_match = npu_op_name == bench_op_name
            return name_match and struct_match
        try:
            name_match = fuzzy_check_op(npu_op_name, bench_op_name)
        except Exception as err:
            logger.warning("%s and %s can not fuzzy match." % (npu_op_name, bench_op_name))
            name_match = False
        return name_match and struct_match

    def match_op(self, npu_queue, bench_queue):
        for b_index, b_op in enumerate(bench_queue[0: -1]):
            if self.check_op(npu_queue[-1], b_op):
                return len(npu_queue) - 1, b_index
        if self.check_op(npu_queue[-1], bench_queue[-1]):
            return len(npu_queue) - 1, len(bench_queue) - 1
        for n_index, n_op in enumerate(npu_queue[0: -1]):
            if self.check_op(n_op, bench_queue[-1]):
                return n_index, len(bench_queue) - 1
        return -1, -1

    def compare_process(self, file_lists):
        npu_json_path, bench_json_path, stack_json_path = file_lists
        npu_json_data = load_json(npu_json_path)
        bench_json_data = load_json(bench_json_path)
        stack_json_data = load_json(stack_json_path) if self.stack_mode else None

        if self.fuzzy_match:
            logger.warning("This task uses fuzzy matching, which may affect the accuracy of the comparison.")

        npu_ops_queue = []
        bench_ops_queue = []
        result = []

        ops_npu_iter = iter(npu_json_data['data'])
        ops_bench_iter = iter(bench_json_data['data'])
        read_err_npu = True
        read_err_bench = True
        last_npu_ops_len = 0
        last_bench_ops_len = 0

        npu_api_nums = len(npu_json_data['data'])
        progress_bar = tqdm(total=npu_api_nums, desc="API/Module Read Progress", unit="item", ncols=100)

        while True:
            if not read_err_npu and not read_err_bench:
                break
            try:
                last_npu_ops_len = len(npu_ops_queue)
                op_name_npu = next(ops_npu_iter)
                check_op_str_pattern_valid(op_name_npu)
                npu_merge_list = self.gen_merge_list(npu_json_data, op_name_npu, stack_json_data)
                if npu_merge_list:
                    npu_ops_queue.append(npu_merge_list)
            except StopIteration:
                read_err_npu = False
            try:
                last_bench_ops_len = len(bench_ops_queue)
                op_name_bench = next(ops_bench_iter)
                check_op_str_pattern_valid(op_name_bench)
                bench_merge_list = self.gen_merge_list(bench_json_data, op_name_bench, stack_json_data)
                if bench_merge_list:
                    bench_ops_queue.append(bench_merge_list)
            except StopIteration:
                read_err_bench = False

            progress_bar.update(1)

            # merge all boolean expressions
            both_empty = not npu_ops_queue and not bench_ops_queue
            no_change = (len(npu_ops_queue) == last_npu_ops_len) and (len(bench_ops_queue) == last_bench_ops_len)
            if both_empty or no_change:
                continue

            # APIs in NPU and Bench models unconsistent judgment
            if bool(npu_ops_queue) ^ bool(bench_ops_queue):
                logger.info("Please check whether the number and calls of APIs in NPU and Bench models are consistent.")
                break

            n_match_point, b_match_point = self.match_op(npu_ops_queue, bench_ops_queue)

            # 如果没有匹配到，数据放到队列中，跳过，直到后面匹配到，把匹配之前的api放到不匹配中
            if n_match_point == -1 and b_match_point == -1:
                continue

            n_match_data = npu_ops_queue[n_match_point]
            b_match_data = bench_ops_queue[b_match_point]
            un_match_data = npu_ops_queue[0: n_match_point]
            for npu_data in un_match_data:
                get_un_match_accuracy(result, npu_data, self.dump_mode)
            get_accuracy(result, n_match_data, b_match_data, self.dump_mode)
            del npu_ops_queue[0: n_match_point + 1]
            del bench_ops_queue[0: b_match_point + 1]
        progress_bar.close()
        if npu_ops_queue:
            for npu_data in npu_ops_queue:
                get_un_match_accuracy(result, npu_data, self.dump_mode)

        result_df = self.make_result_table(result)
        return result_df

    def merge_data(self, json_data, stack_json_data):
        ops_all = {}
        for op_name in json_data.get('data', {}):
            merge_list = self.gen_merge_list(json_data, op_name, stack_json_data)
            if merge_list:
                struct_to_index_mapping = {
                    CompareConst.INPUT_STRUCT: 0,
                    CompareConst.OUTPUT_STRUCT: 0,
                    CompareConst.PARAMS_STRUCT: 0,
                    CompareConst.PARAMS_GRAD_STRUCT: 0
                }

                op_name_list = merge_list.get(CompareConst.OP_NAME)
                summary_list = merge_list.get(Const.SUMMARY)
                data_name_list = merge_list.get('data_name')
                op_name_reorder, summary_reorder, data_name_reorder = reorder_op_x_list(op_name_list,
                                                                                        summary_list,
                                                                                        data_name_list)
                for index, op_full_name in enumerate(op_name_reorder):
                    data_name = data_name_reorder[index] if data_name_reorder else None

                    _, state = get_name_and_state(op_full_name)
                    struct_key = CompareConst.STATE_TO_STRUCT_MAPPING.get(state)
                    if not struct_key:
                        continue
                    ops_all[op_full_name] = {
                        CompareConst.STRUCT: safe_get_value(merge_list, struct_to_index_mapping.get(struct_key),
                                                            "merge_list", key=struct_key),
                        CompareConst.SUMMARY: safe_get_value(summary_reorder, index, "summary_reorder"),
                        'data_name': data_name,
                        'stack_info': merge_list.get('stack_info')
                    }
                    struct_to_index_mapping[struct_key] += 1
        return ops_all

    def get_accuracy(self, npu_ops_all, bench_ops_all):
        result = []
        bench_ops_all[CompareConst.N_A] = self._generate_na_data(bench_ops_all)
        for ms_op_name, bench_op_name in self.data_mapping_dict.items():
            check_op_str_pattern_valid(ms_op_name)
            check_op_str_pattern_valid(bench_op_name)
            if ms_op_name in npu_ops_all and bench_op_name in bench_ops_all:
                npu_stack_info = npu_ops_all.get(ms_op_name).get("stack_info", None)
                bench_stack_info = bench_ops_all.get(bench_op_name).get("stack_info", None)
                has_stack = npu_stack_info and bench_stack_info
                if self.dump_mode == Const.MD5:
                    result.append(self.get_result_md5_compare(ms_op_name, bench_op_name, npu_ops_all,
                                                              bench_ops_all, has_stack, npu_stack_info))
                    continue

                npu_struct = npu_ops_all.get(ms_op_name).get('struct', [])
                bench_struct = bench_ops_all.get(bench_op_name).get('struct', [])

                if len(npu_struct) < 2 or len(bench_struct) < 2:
                    logger.error(
                        f"The length of npu_struct and bench_struct must be >= 2, "
                        f"but got npu_struct={len(npu_struct)} and bench_struct={len(bench_struct)}. "
                        f"Please check!"
                    )
                    raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR)

                base_result_item = [
                    ms_op_name, bench_op_name,
                    npu_struct[0],
                    bench_struct[0],
                    npu_struct[1],
                    bench_struct[1]
                ]

                if self.dump_mode == Const.SUMMARY:
                    result_item = base_result_item + [" "] * 8  # 8个统计量数据情况的比对指标
                else:
                    result_item = base_result_item + [" "] * 6  # 6个真实数据情况的比对指标

                npu_summary_data = npu_ops_all.get(ms_op_name).get("summary")
                result_item.extend(npu_summary_data)
                bench_summary_data = bench_ops_all.get(bench_op_name).get("summary")
                result_item.extend(bench_summary_data)
                if self.dump_mode == Const.SUMMARY:
                    self.calculate_summary_data(npu_summary_data, bench_summary_data, result_item)
                else:
                    result_item.append(CompareConst.ACCURACY_CHECK_YES)
                    result_item.append("")
                if has_stack:
                    result_item.extend(npu_stack_info)
                else:
                    result_item.append(CompareConst.NONE)
                if self.dump_mode == Const.ALL:
                    ms_data_name = npu_ops_all.get(ms_op_name).get("data_name", None)
                    pt_data_name = bench_ops_all.get(bench_op_name).get("data_name", None)
                    result_item.append([ms_data_name, pt_data_name])
                result.append(result_item)
                logger.info(f"{ms_op_name}, {bench_op_name} compared.")
            elif ms_op_name not in npu_ops_all:
                logger.warning(f'Can not find npu op name : `{ms_op_name}` in npu dump json file.')
            elif bench_op_name not in npu_ops_all:
                logger.warning(f'Can not find bench op name : `{bench_op_name}` in bench dump json file.')
        return result

    def compare_process_custom(self, file_lists):
        npu_json_path, bench_json_path, stack_json_path = file_lists
        npu_json_data = load_json(npu_json_path)
        bench_json_data = load_json(bench_json_path)
        stack_json_data = load_json(stack_json_path) if self.stack_mode else None
        npu_ops_all = self.merge_data(npu_json_data, stack_json_data)
        bench_ops_all = self.merge_data(bench_json_data, stack_json_data)

        result = self.get_accuracy(npu_ops_all, bench_ops_all)
        result_df = self.make_result_table(result)
        return result_df

    def compare_by_op(self, npu_op_name, bench_op_name, op_name_mapping_dict, input_param):
        """
        :param npu_op_name: excel中的NPU_Name，例如：MintFunctional.conv2d.0.forward.input.3.0
        :param bench_op_name: excel中的Bench_Name，例如：Functional.conv2d.0.forward.input.3.0
        :param op_name_mapping_dict: op_name和npy或pt文件的映射关系
        :param input_param: npu_json_path/bench_json_path/stack_json_path等参数
        :return: result_list，包含余弦相似度、最大绝对误差、最大相对误差、千分之一误差率、千分之五误差率和错误信息
        用于读取excel中的NPU_Name和Bench_Name，根据映射关系找到npy或pt文件，然后读取文件中的数据进行比较，计算余弦相似度、欧式距离
        最大绝对误差、最大相对误差、千分之一误差率、千分之五误差率并生成错误信息
        """
        error_file, relative_err, error_flag = None, None, False

        data_name_pair = op_name_mapping_dict.get(npu_op_name)
        npu_data_name = data_name_pair[0]
        bench_data_name = data_name_pair[1]

        if str(npu_data_name) == '-1':  # 没有npu真实数据
            n_value, b_value, error_flag = CompareConst.READ_NONE, CompareConst.READ_NONE, True
        elif str(bench_data_name) == '-1':  # 没有bench真实数据
            n_value, b_value, error_flag = CompareConst.READ_NONE, CompareConst.READ_NONE, True
            error_file = 'no_bench_data'
        else:
            npu_dir = input_param.get("npu_dump_data_dir")
            bench_dir = input_param.get("bench_dump_data_dir")
            try:
                frame_name = getattr(self, "frame_name")
                read_npy_data = getattr(self, "read_npy_data")
                if frame_name == "MSComparator":
                    n_value = read_npy_data(npu_dir, npu_data_name)
                    if self.cross_frame:
                        b_value = read_npy_data(bench_dir, bench_data_name, load_pt_file=True)
                    else:
                        b_value = read_npy_data(bench_dir, bench_data_name)
                else:
                    n_value = read_npy_data(npu_dir, npu_data_name)
                    b_value = read_npy_data(bench_dir, bench_data_name)
            except IOError as error:
                error_file = error.filename
                n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
                error_flag = True
            except (FileCheckException, CompareException):
                error_file = npu_data_name
                n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
                error_flag = True

        # 通过n_value, b_value同时得到错误标志和错误信息
        n_value, b_value, error_flag, err_msg = get_error_flag_and_msg(n_value, b_value,
                                                                       error_flag=error_flag, error_file=error_file)

        result_list, err_msg = compare_ops_apply(n_value, b_value, error_flag, err_msg)

        if self.fuzzy_match and npu_op_name != bench_op_name and bench_op_name != CompareConst.N_A:
            err_msg += " Fuzzy matching data, the comparison accuracy may be affected."
        result_list.append(err_msg)
        return result_list

    def compare_core(self, input_param, output_path, **kwargs):
        """
        Compares data from multiple JSON files and generates a comparison report.

        Args:
            input_param (dict): A dictionary containing paths to JSON files ("npu_path", "bench_path",
                                "stack_path").
            output_path (str): The path where the output Excel report will be saved.
            **kwargs: Additional keyword arguments including:
            - stack_mode (bool, optional): Enables stack mode comparison. Defaults to False.
            - auto_analyze (bool, optional): If True, triggers automatic analysis after comparison. Defaults to True.
            - suffix (str, optional): Suffix to append to the output file name. Defaults to ''.
            - fuzzy_match (bool, optional): Enables fuzzy matching during comparison. Defaults to False.
            - dump_mode (str): ALL, SUMMARY, MD5.

        Returns:
        """
        # get kwargs or set default value
        suffix = kwargs.get('suffix', '')

        logger.info("Please check whether the input data belongs to you. If not, there may be security risks.")
        file_name = add_time_with_xlsx("compare_result" + suffix)
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        if os.path.exists(file_path):
            logger.warning(f"{file_path} will be deleted.")
            remove_path(file_path)
        highlight_dict = {"red_rows": set(), "yellow_rows": set(), "red_lines": [], "yellow_lines": []}

        npu_json = input_param.get("npu_json_path")
        bench_json = input_param.get("bench_json_path")
        stack_json = input_param.get("stack_json_path")
        if self.data_mapping:
            result_df = self.compare_process_custom([npu_json, bench_json, stack_json])
        else:
            result_df = self.compare_process([npu_json, bench_json, stack_json])

        if not result_df.values.tolist():
            logger.warning("Can`t match any op.")
            return

        if self.dump_mode == Const.ALL:
            result_df = self.do_multi_process(input_param, result_df)

        find_compare_result_error_rows(result_df, highlight_dict, self.dump_mode)
        highlight_rows_xlsx(result_df, highlight_dict, file_path)

        if self.auto_analyze:
            advisor = Advisor(result_df, output_path, suffix)
            advisor.analysis()

        print_compare_ends_info()

    def compare_ops(self, idx, dump_path_dict, result_df, lock, input_param):
        cos_result = []
        euc_dist_result = []
        max_err_result = []
        max_relative_err_result = []
        one_thousand_err_ratio_result = []
        five_thousand_err_ratio_result = []
        err_mess = []

        is_print_compare_log = input_param.get("is_print_compare_log")

        for i in range(len(result_df)):
            npu_op_name = result_df.iloc[i, 0]
            bench_op_name = result_df.iloc[i, 1]
            if is_print_compare_log:
                logger.info("start compare: {}".format(npu_op_name))

            cos_sim, euc_dist, max_abs_err, max_relative_err, one_thousand_err_ratio, five_thousand_err_ratio, err_msg \
                = self.compare_by_op(npu_op_name, bench_op_name, dump_path_dict, input_param)

            if is_print_compare_log:
                logger.info(
                    "[{}] Compare result: cosine {}, max_abs_err {}, max_relative_err {}, {}, \
                    one_thousand_err_ratio {}, "
                    "five_thousand_err_ratio {}".format(npu_op_name, cos_sim, max_abs_err, max_relative_err,
                                                        err_msg, one_thousand_err_ratio, five_thousand_err_ratio))
            cos_result.append(cos_sim)
            euc_dist_result.append(euc_dist)
            max_err_result.append(max_abs_err)
            max_relative_err_result.append(max_relative_err)
            one_thousand_err_ratio_result.append(one_thousand_err_ratio)
            five_thousand_err_ratio_result.append(five_thousand_err_ratio)
            err_mess.append(err_msg)

        cr = ComparisonResult(
            cos_result=cos_result,
            euc_dist_result=euc_dist_result,
            max_err_result=max_err_result,
            max_relative_err_result=max_relative_err_result,
            one_thousand_err_ratio_result=one_thousand_err_ratio_result,
            five_thousand_err_ratio_result=five_thousand_err_ratio_result,
            err_msgs=err_mess
        )

        return _save_cmp_result(idx, cr, result_df, lock)

    def do_multi_process(self, input_param, result_df):
        try:
            result_df = _handle_multi_process(self.compare_ops, input_param, result_df,
                                              multiprocessing.Manager().RLock())
            return result_df
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e


class ParseData:
    def __init__(self, mode_config: ModeConfig):
        self.mode_config = mode_config

    def parse(self, file_list):
        npu_json_path, bench_json_path, stack_json_path = file_list
        npu_json_data = load_json(npu_json_path)
        bench_json_data = load_json(bench_json_path)
        stack_json_data = load_json(stack_json_path) if self.mode_config.stack_mode else None

        # parse json data and generate df
        npu_df = self.gen_data_df(npu_json_data, stack_json_data)
        bench_df = self.gen_data_df(bench_json_data, stack_json_data)

        return npu_df, bench_df

    def gen_data_df(self, data_json, stack_json_data):
        result = {
            CompareConst.OP_NAME: [],
            Const.DTYPE: [],
            Const.SHAPE: [],
            Const.SUMMARY: [],
            Const.STACK_INFO: []
        }
        if self.mode_config.dump_mode == Const.ALL:
            result['data_name'] = []
        elif self.mode_config.dump_mode == Const.MD5:
            result[Const.MD5] = []

        api_nums = len(data_json['data'])
        progress_bar = tqdm(total=api_nums, desc="API/Module Read Progress", unit="api/module", ncols=100)

        # 从json中循环解析API数据，遍历所有API
        for data_name in data_json['data']:
            check_op_str_pattern_valid(data_name)
            merge_list = self.gen_merge_list(data_json, data_name, stack_json_data)
            if not merge_list:
                continue

            op_name_list = merge_list.get(CompareConst.OP_NAME)
            summary_list = merge_list.get(Const.SUMMARY)
            data_name_list = merge_list.get('data_name')
            op_name_reorder, summary_reorder, data_name_reorder = reorder_op_x_list(op_name_list,
                                                                                    summary_list,
                                                                                    data_name_list)
            # 遍历单个API的所有item
            for index, op_name in enumerate(op_name_reorder):
                result[CompareConst.OP_NAME].append(op_name)
                if (CompareConst.INPUT_PATTERN in op_name) or (CompareConst.KWARGS_PATTERN in op_name):
                    struct = merge_list[CompareConst.INPUT_STRUCT].pop(0)
                elif CompareConst.OUTPUT_PATTERN in op_name:
                    struct = merge_list[CompareConst.OUTPUT_STRUCT].pop(0)
                elif CompareConst.PARAMS_PATTERN in op_name:
                    struct = merge_list[CompareConst.PARAMS_STRUCT].pop(0)
                else:
                    struct = merge_list[CompareConst.PARAMS_GRAD_STRUCT].pop(0)
                result[Const.DTYPE].append(struct[0])
                result[Const.SHAPE].append(struct[1])
                if self.mode_config.dump_mode == Const.MD5:
                    result[Const.MD5].append(struct[2])
                result[Const.SUMMARY].append(summary_reorder.pop(0))
                result[Const.STACK_INFO].append(
                    merge_list[Const.STACK_INFO][0] if index == 0 and self.mode_config.stack_mode else None)
                if self.mode_config.dump_mode == Const.ALL:
                    result['data_name'].append(data_name_reorder.pop(0))

            progress_bar.update(1)
        progress_bar.close()
        return pd.DataFrame(result)

    def gen_merge_list(self, json_data, op_name, stack_json_data):
        op_data = json_data['data'][op_name]
        check_dump_json_str(op_data, op_name)
        op_parsed_list = read_op(op_data, op_name)

        if self.mode_config.stack_mode:
            stack_info = stack_json_data.get(op_name)
            if stack_info is not None:
                check_stack_json_str(stack_info, op_name)
            # append only when stack_mode is True,
            op_parsed_list.append({
                'full_op_name': op_name,
                'full_info': stack_info
            })

        merge_list = merge_tensor(op_parsed_list, self.mode_config.dump_mode)
        return merge_list


class ProcessDf:
    def __init__(self, mode_config: ModeConfig, mapping_config: MappingConfig, mapping_dict: MappingDict):
        self.mode_config = mode_config
        self.mapping_config = mapping_config
        self.mapping_dict = mapping_dict

    @staticmethod
    def get_api_name(api_list):
        try:
            api_name = api_list[0] + Const.SEP + api_list[1]
        except IndexError as error:
            logger.error('Failed to retrieve API name, please check if the dump data is reasonable')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        return api_name

    def process_compare_key_and_shape(self, npu_df, bench_df):
        npu_df = self.assign_npu_df_compare_key(npu_df, bench_df)
        npu_df[CompareConst.CMP_SHAPE] = npu_df[Const.SHAPE]
        bench_df[CompareConst.CMP_KEY] = bench_df[CompareConst.OP_NAME]
        bench_df[CompareConst.CMP_SHAPE] = bench_df[Const.SHAPE]
        return npu_df, bench_df

    def assign_npu_df_compare_key(self, npu_df, bench_df):
        """
        处理 npu_df 的 COMPARE_KEY 赋值逻辑

        :param npu_df: DataFrame，NPU 对比数据
        :param bench_df: DataFrame，Bench 对比数据
        :return: compare_key(name)处理后的 npu_df
        """
        # 处理api_mapping映射
        if self.mapping_config.api_mapping:
            # 如果用户不传api_mapping.yaml，先使用内置api_mapping.yaml替换npu_op_name
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_internal_api_mapping)
            # 如果用户传入api_mapping.yaml，再使用传入api_mapping.yaml进一步替换npu_op_name
            if isinstance(self.mapping_config.api_mapping, str):
                self.modify_compare_data_with_user_mapping(npu_df, bench_df)
        # 处理cell_mapping映射
        elif self.mapping_config.cell_mapping:
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_cell_mapping)
        # 处理data_mapping映射
        elif self.mapping_config.data_mapping:
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME].apply(self.process_data_mapping)
        else:
            npu_df[CompareConst.CMP_KEY] = npu_df[CompareConst.OP_NAME]
        return npu_df

    def process_internal_api_mapping(self, npu_op_name):
        # get api name & class name from op_name
        ms_api_name = self.get_api_name(npu_op_name.split(Const.SEP))
        class_name = ms_api_name.split(Const.SEP)[0]
        if class_name == "Mint":
            return npu_op_name.replace("Mint", "Torch")
        elif class_name == "MintFunctional":
            return npu_op_name.replace("MintFunctional", "Functional")
        elif self.mapping_dict.ms_to_pt_mapping.get(ms_api_name):
            return npu_op_name.replace(ms_api_name, self.mapping_dict.ms_to_pt_mapping.get(ms_api_name))
        else:
            return npu_op_name

    def modify_compare_data_with_user_mapping(self, npu_df, bench_df):
        def gen_input_compare_key(pattern, term):
            is_unmatched = True
            for i, prefix in enumerate(mapping_dict.get(f'ms_{term}')):
                if op_name.split(pattern)[1].startswith(str(prefix)):
                    npu_df.loc[index, CompareConst.CMP_KEY] = (
                        op_name.replace(pattern + str(prefix),
                                        pattern + str(mapping_dict.get(f'pt_{term}')[i])))
                    is_unmatched = False
            return is_unmatched

        ms_api_indices_dict = self.get_api_indices_dict(npu_df)
        pt_api_indices_dict = self.get_api_indices_dict(bench_df)

        for mapping_dict in self.mapping_dict.api_mapping_dict:
            all_length_equal = True
            for k1, k2 in CompareConst.API_MAPPING_KEYS_TO_COMPARE:
                if len(mapping_dict.get(k1, [])) != len(mapping_dict.get(k2, [])):
                    all_length_equal = False
            if not all_length_equal:
                logger.warning('The user-defined mapping table is incorrect,\
                                make sure that the number of parameters is equal')
                continue

            ms_api, pt_api = mapping_dict.get('ms_api'), mapping_dict.get('pt_api')
            if ms_api not in ms_api_indices_dict or pt_api not in pt_api_indices_dict:
                continue
            for index in ms_api_indices_dict.get(ms_api):
                op_name = npu_df.loc[index, CompareConst.OP_NAME].replace(ms_api, pt_api, 1)
                if CompareConst.INPUT_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.INPUT_PATTERN, 'args')
                elif CompareConst.KWARGS_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.KWARGS_PATTERN, 'args')
                elif CompareConst.OUTPUT_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.OUTPUT_PATTERN, 'output')
                elif CompareConst.PARAMS_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.PARAMS_PATTERN, 'parameters')
                elif CompareConst.PARAMS_GRAD_PATTERN in op_name:
                    is_abandoned = gen_input_compare_key(CompareConst.PARAMS_GRAD_PATTERN, 'parameters_grad')
                else:
                    logger.error(f'Excepted op_name: {op_name}')
                    raise CompareException(CompareException.INVALID_DATA_ERROR)
                if is_abandoned:
                    npu_df.loc[index, CompareConst.CMP_KEY] = op_name + 'abandoned'

    def get_api_indices_dict(self, op_name_df):
        """
        生成多个api对应的各自的所有的input、output等的index的键值对字典
        示例：
        {'Functional.conv2d': [0, 1, 2, 3],
        'Functional.batch_norm': [4, 5, 6, 7, 8]
        }
        """
        api_indices_dict = defaultdict(list)
        for op_index, name in enumerate(op_name_df[CompareConst.OP_NAME]):
            api_name = self.get_api_name(name.split(Const.SEP))
            api_indices_dict[api_name].append(op_index)
        return api_indices_dict

    def process_cell_mapping(self, npu_op_name):
        if not npu_op_name:
            return CompareConst.N_A
        param_grad_flag = Const.PARAMS_GRAD in npu_op_name.split(Const.SEP)
        if not param_grad_flag and not re.search(Const.REGEX_FORWARD_BACKWARD, npu_op_name):
            return CompareConst.N_A
        npu_op_name = npu_op_name.replace("Cell", "Module", 1)
        if self.mapping_dict.cell_mapping_dict:
            # get cell name & class name from op_name
            # Cell.fc1.Dense.forward.0.input.0
            cell_name = re.split(r'\.(?:forward|backward|parameters_grad)\.', npu_op_name.split(Const.SEP, 1)[-1])[0]
            if cell_name in self.mapping_dict.cell_mapping_dict:
                npu_op_name = npu_op_name.replace(cell_name, self.mapping_dict.cell_mapping_dict[cell_name], 1)
        return npu_op_name

    def process_data_mapping(self, npu_op_name):
        return self.mapping_dict.data_mapping_dict.get(npu_op_name, npu_op_name)


class Match:
    def __init__(self, mode_config: ModeConfig, mapping_config: MappingConfig, cross_frame):
        self.mode_config = mode_config
        self.mapping_config = mapping_config
        self.cross_frame = cross_frame

    @staticmethod
    def put_unmatched_in_table(match_result, npu_op_item):
        npu_columns = npu_op_item.index.tolist()[:-2]
        new_columns = [name[:-1] + 'y' for name in npu_columns]
        na_series = pd.Series([CompareConst.N_A] * len(new_columns), index=new_columns)
        new_result_item = pd.concat([npu_op_item, na_series]).to_frame().T
        new_result_item.columns = CompareConst.MATCH_RESULT_COLUMNS
        match_result = pd.concat([match_result, new_result_item])
        return match_result

    @staticmethod
    def put_matched_in_table(match_result, npu_op_item, bench_op_item):
        head_len = len(CompareConst.MATCH_RESULT_COLUMNS)
        new_result_item = pd.concat([npu_op_item, bench_op_item]).head(head_len).to_frame().T
        new_result_item.columns = CompareConst.MATCH_RESULT_COLUMNS
        match_result = pd.concat([match_result, new_result_item])
        return match_result

    @staticmethod
    def rename_api(op_name):
        """
        原api： {api_type}.{api_name}.{API调用次数}.{前向反向}.{input/output}.{参数序号}
        rename后： {api_type}.{api_name}.{API调用次数}.{input/output}.{参数序号}
        """
        process = Const.FORWARD if Const.FORWARD in op_name else Const.BACKWARD
        name_split = op_name.split(process)
        try:
            torch_func_index, in_out = name_split[0], name_split[1]
        except IndexError as error:
            logger.error(f'{op_name} can not be split with {process}, please check!')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from error
        torch_func_split = torch_func_index.rsplit(Const.SEP, 2)
        torch_func = str(torch_func_split[0]) + Const.SEP + process + str(in_out)
        return torch_func

    def check_op_item(self, npu_op_item, bench_op_item):
        name_match = self.rename_api(npu_op_item[CompareConst.CMP_KEY]) == self.rename_api(
            bench_op_item[CompareConst.CMP_KEY])
        shape_match = npu_op_item[CompareConst.CMP_SHAPE] == bench_op_item[CompareConst.CMP_SHAPE]
        if name_match and shape_match:
            return True
        else:
            npu_op_name = npu_op_item[CompareConst.OP_NAME]
            bench_op_name = bench_op_item[CompareConst.OP_NAME]
            check_op_str_pattern_valid(npu_op_name)
            check_op_str_pattern_valid(bench_op_name)
            logger.warning(f"{npu_op_name} and {bench_op_name} can not fuzzy match")
            return False

    def match_api_infos(self, npu_df, bench_df):
        """
        正常匹配和模糊匹配
        """
        if self.mapping_config.data_mapping:
            match_result = pd.merge(npu_df, bench_df, on=[CompareConst.CMP_KEY], how='outer')
        elif not self.mode_config.fuzzy_match:
            match_result = pd.merge(npu_df, bench_df, on=[CompareConst.CMP_KEY, CompareConst.CMP_SHAPE],
                                    how='outer')
        else:
            match_result = self.process_fuzzy_match(npu_df, bench_df)
        return match_result

    def process_fuzzy_match(self, npu_df, bench_df):
        """
        模糊匹配通过循环方式匹配api
        """
        npu_ops_queue = []
        bench_ops_queue = []
        match_result = pd.DataFrame(columns=CompareConst.MATCH_RESULT_COLUMNS)

        max_len = max(len(npu_df), len(bench_df))
        min_len = min(len(npu_df), len(bench_df))
        for i in range(max_len):
            if i < min_len:
                npu_ops_queue.append(npu_df.iloc[i])
                bench_ops_queue.append(bench_df.iloc[i])
            else:
                try:
                    npu_ops_queue.append(npu_df.iloc[i])
                except IndexError:
                    pass
                try:
                    bench_ops_queue.append(bench_df.iloc[i])
                except IndexError:
                    pass

            # 如果append之后queue状态不一致，则判断结束
            if bool(npu_ops_queue) ^ bool(bench_ops_queue):
                break

            npu_match_point, bench_match_point = self.match_op(npu_ops_queue, bench_ops_queue)

            # 如果没有匹配到，数据放到队列中，跳过。直到后面匹配到，把匹配之前的api放到不匹配中
            if npu_match_point == -1 and bench_match_point == -1:
                continue

            npu_op_item = npu_ops_queue[npu_match_point]
            bench_op_item = bench_ops_queue[bench_match_point]
            unmatched_data = npu_ops_queue[0: npu_match_point]
            for op_item in unmatched_data:
                match_result = self.put_unmatched_in_table(match_result, op_item)
            match_result = self.put_matched_in_table(match_result, npu_op_item, bench_op_item)
            del npu_ops_queue[0: npu_match_point + 1]
            del bench_ops_queue[0: bench_match_point + 1]

        if npu_ops_queue:
            for op_item in npu_ops_queue:
                match_result = self.put_unmatched_in_table(match_result, op_item)

        match_result.reset_index(drop=True, inplace=True)
        return match_result

    def match_op(self, npu_queue, bench_queue):
        for b_index, b_op in enumerate(bench_queue[0: -1]):
            if self.check_op_item(npu_queue[-1], b_op):
                return len(npu_queue) - 1, b_index
        if self.check_op_item(npu_queue[-1], bench_queue[-1]):
            return len(npu_queue) - 1, len(bench_queue) - 1
        for n_index, n_op in enumerate(npu_queue[0: -1]):
            if self.check_op_item(n_op, bench_queue[-1]):
                return n_index, len(bench_queue) - 1
        return -1, -1

    def gen_dtype_condition(self, match_result):
        """
        dtype匹配条件为npu、bench的dtype一致或属于规定的映射关系
        """
        # 如果使用了data_mapping，不校验dtype，返回全True的DataFrame
        if self.mapping_config.data_mapping:
            return pd.Series(True, index=match_result.index)

        npu_dtype = match_result['dtype_x']
        bench_dtype = match_result['dtype_y']
        npu_dtype = self.process_cross_frame_dtype(npu_dtype)
        bench_dtype = self.process_cross_frame_dtype(bench_dtype)

        equal_condition = npu_dtype == bench_dtype
        match_condition = (
                (npu_dtype.isin(CompareConst.DTYPE_MATCH_GROUPS[0]) & bench_dtype.isin(
                    CompareConst.DTYPE_MATCH_GROUPS[0])) |
                (npu_dtype.isin(CompareConst.DTYPE_MATCH_GROUPS[1]) & bench_dtype.isin(
                    CompareConst.DTYPE_MATCH_GROUPS[1]))
        )
        return equal_condition | match_condition

    def process_cross_frame_dtype(self, dtype):
        if self.cross_frame:
            dtype = dtype.map(cross_dtype_mapping_2).fillna(dtype)
        return dtype


class CreateTable:
    def __init__(self, mode_config: ModeConfig):
        self.mode_config = mode_config

    @staticmethod
    def process_data_name(result):
        result['data_name_x'] = result.apply(lambda row: [row['data_name_x'], row['data_name_y']], axis=1)
        return result

    @staticmethod
    def set_summary(summary):
        if summary == CompareConst.N_A:
            return [CompareConst.N_A] * 4  # 4为统计值个数
        summary_list = []
        for i in summary:
            if str(i).lower() == 'nan':
                summary_list.append(CompareConst.NAN)
            else:
                summary_list.append(i)
        return summary_list

    def make_result_df(self, result):
        # get header
        header = CompareConst.HEAD_OF_COMPARE_MODE[self.mode_config.dump_mode][:]
        if self.mode_config.stack_mode:
            header.append(CompareConst.STACK)
        if self.mode_config.dump_mode == Const.ALL:
            header.append(CompareConst.DATA_NAME)
            result = self.process_data_name(result)

        # rename match_result columns
        result.rename(columns={'op_name_x': CompareConst.NPU_NAME,
                               'op_name_y': CompareConst.BENCH_NAME,
                               'dtype_x': CompareConst.NPU_DTYPE,
                               'dtype_y': CompareConst.BENCH_DTYPE,
                               'shape_x': CompareConst.NPU_SHAPE,
                               'shape_y': CompareConst.BENCH_SHAPE,
                               'md5_x': CompareConst.NPU_MD5,
                               'md5_y': CompareConst.BENCH_MD5,
                               'data_name_x': CompareConst.DATA_NAME,
                               'stack_info_x': CompareConst.STACK}, inplace=True)

        # process summary data
        npu_summary = [CompareConst.NPU_MAX, CompareConst.NPU_MIN, CompareConst.NPU_MEAN, CompareConst.NPU_NORM]
        bench_summary = [CompareConst.BENCH_MAX, CompareConst.BENCH_MIN, CompareConst.BENCH_MEAN,
                         CompareConst.BENCH_NORM]
        result[npu_summary] = result['summary_x'].apply(self.set_summary).tolist()
        result[bench_summary] = result['summary_y'].apply(self.set_summary).tolist()

        result_df = pd.DataFrame(columns=header)
        for h in header:
            if h in result.columns:
                result_df[h] = result[h]
        return result_df, header


class CalcStatsDiff:
    def __init__(self, mode_config: ModeConfig):
        self.mode_config = mode_config

    @staticmethod
    def type_check(val):
        """
        检查是否为数值或字符串形式的nan, 如果是返回True
        """
        check_series = pd.Series(False, index=val.index)
        val_str = val.astype(str)
        check_series[pd.to_numeric(val_str, errors='coerce').notna() | val_str.str.lower().eq('nan')] = True
        return check_series

    @staticmethod
    def get_number(val):
        return pd.to_numeric(val.astype(str), errors='coerce')

    def calc_summary_diff(self, result_df, cond_no_bench, stats_index: str):
        npu_val = result_df['NPU ' + stats_index]
        bench_val = result_df['Bench ' + stats_index]
        diff_name = stats_index.capitalize() + ' diff'
        rel_err_name = ('norm' if stats_index == 'l2norm' else stats_index).capitalize() + 'RelativeErr'

        # npu、bench中统计量均为数字或nan
        cond_num_nan = self.type_check(npu_val) & self.type_check(bench_val)

        # 如果统计量不是数字或nan，就赋值统计量差异为N/A
        result_df.loc[~cond_num_nan, [diff_name, rel_err_name]] = CompareConst.N_A
        cond_valid_stat = ~cond_no_bench & cond_num_nan  # 有效统计条件：bench_name不是N/A，并且NPU和bench的统计量都是数字或nan
        result_df.loc[cond_valid_stat, diff_name] = self.get_number(npu_val) - self.get_number(bench_val)

        cond_diff_nan = result_df[diff_name].isna()  # 统计量差异是nan
        cond_nan_diff = cond_valid_stat & cond_diff_nan
        result_df.loc[cond_nan_diff, [diff_name, rel_err_name]] = CompareConst.NAN

        cond_not_nan_diff = cond_valid_stat & ~cond_diff_nan
        condition_pt_zero = bench_val == 0
        result_df.loc[cond_not_nan_diff & condition_pt_zero, rel_err_name] = CompareConst.N_A

        # 相对误差转成百分比字符串
        cond_ref_err = cond_not_nan_diff & ~condition_pt_zero
        result_df.loc[cond_ref_err, rel_err_name] = (
                result_df.loc[cond_ref_err, diff_name] / bench_val[cond_ref_err] * 100)
        result_df.loc[cond_ref_err, rel_err_name] = (result_df.loc[cond_ref_err, rel_err_name].abs().astype(str) + '%')

        magnitude = self.get_number(result_df[diff_name]).abs() / (pd.Series(
            np.maximum(self.get_number(npu_val), self.get_number(bench_val))).abs() + CompareConst.EPSILON)
        return magnitude > CompareConst.MAGNITUDE

    def calc_accuracy(self, result_df, header):
        # bench name N/A represents no bench data, err_msg adds "No bench data matched."
        condition_no_bench = result_df[CompareConst.BENCH_NAME] == CompareConst.N_A
        result_df[condition_no_bench] = result_df[condition_no_bench].fillna(CompareConst.N_A)
        result_df.loc[condition_no_bench, CompareConst.ERROR_MESSAGE] = CompareConst.NO_BENCH

        if self.mode_config.dump_mode == Const.MD5:
            condition_md5_equal = result_df[CompareConst.NPU_MD5] == result_df[CompareConst.BENCH_MD5]
            result_df.loc[condition_md5_equal, CompareConst.RESULT] = CompareConst.PASS
            result_df.loc[~condition_md5_equal & ~condition_no_bench, CompareConst.RESULT] = CompareConst.DIFF
        elif self.mode_config.dump_mode == Const.SUMMARY:
            warning_list = [
                self.calc_summary_diff(result_df, condition_no_bench, stats_index)
                for stats_index in ['max', 'min', 'mean', 'l2norm']
            ]
            warning_flag = pd.DataFrame(warning_list).any()
            result_df.loc[~condition_no_bench, [CompareConst.RESULT, CompareConst.ERROR_MESSAGE]] = ''
            result_df.loc[warning_flag, CompareConst.RESULT] = CompareConst.WARNING
            result_df.loc[warning_flag, CompareConst.ERROR_MESSAGE] = 'Need double check api accuracy.'
        else:
            fill_cols = [CompareConst.COSINE, CompareConst.EUC_DIST,
                         CompareConst.MAX_ABS_ERR, CompareConst.MAX_RELATIVE_ERR,
                         CompareConst.ONE_THOUSANDTH_ERR_RATIO, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO,
                         CompareConst.ERROR_MESSAGE]
            result_df.loc[~condition_no_bench, fill_cols] = ''
            result_df.loc[~condition_no_bench, CompareConst.ACCURACY] = CompareConst.ACCURACY_CHECK_YES

        return result_df[header]


def setup_comparison(input_param, output_path, **kwargs) -> ComparisonConfig:
    """公共的前置处理逻辑，返回封装后的 ComparisonConfig 对象"""
    try:
        config = ComparisonConfig(
            dump_mode='',
            stack_mode=False,
            auto_analyze=kwargs.get('auto_analyze', True),
            fuzzy_match=kwargs.get('fuzzy_match', False),
            data_mapping=kwargs.get('data_mapping', {}),
            suffix=kwargs.get('suffix', ''),
            cell_mapping=kwargs.get('cell_mapping', {}),
            api_mapping=kwargs.get('api_mapping', {}),
            layer_mapping=kwargs.get('layer_mapping', {}),
        )

        set_dump_path(input_param)
        config.dump_mode = get_dump_mode(input_param)

        # set stack_mode and set "stack_json_path" in input_param
        if 'stack_json_path' in input_param:
            config.stack_mode = kwargs.get('stack_mode', False)
        else:
            config.stack_mode = set_stack_json_path(input_param)

        check_configuration_param(config.stack_mode, config.auto_analyze, config.fuzzy_match,
                                  input_param.get('is_print_compare_log', True))
        create_directory(output_path)
        check_compare_param(input_param, output_path, config.dump_mode, config.stack_mode)

        return config

    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        raise CompareException(error.code) from error
