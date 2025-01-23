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

import pandas as pd
from tqdm import tqdm

from msprobe.core.advisor.advisor import Advisor
from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_utils import load_json, remove_path
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException, add_time_with_xlsx, check_op_str_pattern_valid, safe_get_value
from msprobe.core.compare.check import check_dump_json_str, check_graph_mode, check_stack_json_str, \
    check_struct_match, fuzzy_check_op
from msprobe.core.compare.highlight import find_compare_result_error_rows, highlight_rows_xlsx
from msprobe.core.compare.multiprocessing_compute import ComparisonResult, _handle_multi_process, _save_cmp_result
from msprobe.core.compare.npy_compare import compare_ops_apply, get_error_flag_and_msg
from msprobe.core.compare.utils import get_accuracy, get_rela_diff_summary_mode, get_un_match_accuracy, merge_tensor, \
    print_compare_ends_info, read_op, get_name_and_state, reorder_op_x_list


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
                    result_item = base_result_item + [" "] * 8
                else:
                    result_item = base_result_item + [" "] * 5

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
                    result_item.append(npu_ops_all.get(ms_op_name).get("data_name", None))
                result.append(result_item)
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

    def compare_by_op(self, npu_op_name, bench_op_name, op_name_mapping_dict, input_param, bench_data):
        """
        :param npu_op_name: excel中的NPU_Name，例如：MintFunctional.conv2d.0.forward.input.3.0
        :param bench_op_name: excel中的Bench_Name，例如：Functional.conv2d.0.forward.input.3.0
        :param op_name_mapping_dict: op_name和npy或pt文件的映射关系
        :param input_param: npu_json_path/bench_json_path/stack_json_path等参数
        :param bench_data: bench的dump数据中"data"字段
        :return: result_list，包含余弦相似度、最大绝对误差、最大相对误差、千分之一误差率、千分之五误差率和错误信息
        用于读取excel中的NPU_Name和Bench_Name，根据映射关系找到npy或pt文件，然后读取文件中的数据进行比较，计算余弦相似度、
        最大绝对误差、最大相对误差、千分之一误差率、千分之五误差率并生成错误信息
        """
        npu_bench_name_list = op_name_mapping_dict[npu_op_name]
        data_name = safe_get_value(npu_bench_name_list, 1, "npu_bench_name_list")
        error_file, relative_err, error_flag = None, None, False
        bench_data_name = get_bench_data_name(bench_op_name, bench_data)
        if data_name == '-1' or data_name == -1:  # 没有真实数据路径
            n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
            error_flag = True
        elif not bench_data_name:
            n_value, b_value, error_flag = CompareConst.READ_NONE, CompareConst.READ_NONE, True
            error_file = 'no_bench_data'
        else:
            try:
                read_npy_data = getattr(self, "read_npy_data")
                frame_name = getattr(self, "frame_name")
                if frame_name == "MSComparator":
                    n_value = read_npy_data(input_param.get("npu_dump_data_dir"), npu_op_name + Const.NUMPY_SUFFIX)
                    if self.cross_frame:
                        b_value = read_npy_data(input_param.get("bench_dump_data_dir"), bench_data_name,
                                                load_pt_file=True)
                    else:
                        b_value = read_npy_data(input_param.get("bench_dump_data_dir"), bench_data_name)
                else:
                    n_value = read_npy_data(input_param.get("npu_dump_data_dir"), npu_op_name + Const.PT_SUFFIX)
                    b_value = read_npy_data(input_param.get("bench_dump_data_dir"), bench_data_name)
            except IOError as error:
                error_file = error.filename
                n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
                error_flag = True
            except (FileCheckException, CompareException):
                error_file = data_name
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
        max_err_result = []
        max_relative_err_result = []
        err_mess = []
        one_thousand_err_ratio_result = []
        five_thousand_err_ratio_result = []
        is_print_compare_log = input_param.get("is_print_compare_log")
        bench_data = load_json(input_param.get("bench_json_path")).get('data')
        for i in range(len(result_df)):
            npu_op_name = result_df.iloc[i, 0]
            bench_op_name = result_df.iloc[i, 1]
            if is_print_compare_log:
                logger.info("start compare: {}".format(npu_op_name))

            cos_sim, max_abs_err, max_relative_err, one_thousand_err_ratio, five_thousand_err_ratio, err_msg = \
                self.compare_by_op(npu_op_name, bench_op_name, dump_path_dict, input_param, bench_data)

            if is_print_compare_log:
                logger.info(
                    "[{}] Compare result: cosine {}, max_abs_err {}, max_relative_err {}, {}, \
                    one_thousand_err_ratio {}, "
                    "five_thousand_err_ratio {}".format(npu_op_name, cos_sim, max_abs_err, max_relative_err,
                                                        err_msg, one_thousand_err_ratio, five_thousand_err_ratio))
            cos_result.append(cos_sim)
            max_err_result.append(max_abs_err)
            max_relative_err_result.append(max_relative_err)
            err_mess.append(err_msg)
            one_thousand_err_ratio_result.append(one_thousand_err_ratio)
            five_thousand_err_ratio_result.append(five_thousand_err_ratio)

        cr = ComparisonResult(
            cos_result=cos_result,
            max_err_result=max_err_result,
            max_relative_err_result=max_relative_err_result,
            err_msgs=err_mess,
            one_thousand_err_ratio_result=one_thousand_err_ratio_result,
            five_thousand_err_ratio_result=five_thousand_err_ratio_result
        )

        return _save_cmp_result(idx, cr, result_df, lock)

    def do_multi_process(self, input_parma, result_df):
        try:
            result_df = _handle_multi_process(self.compare_ops, input_parma, result_df,
                                              multiprocessing.Manager().RLock())
            return result_df
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e


def get_bench_data_name(bench_op_name, bench_data):
    bench_name_list = re.split(r'\.(input|output|kwargs|parameters|parameters_grad)\.', bench_op_name)
    if len(bench_name_list) > 1 and bench_name_list[1] == Const.PARAMS_GRAD:
        bench_data_bundle = bench_data.get(bench_name_list[0] + Const.SEP + bench_name_list[1], {})
    else:
        bench_data_bundle = bench_data.get(bench_name_list[0], {})
    if not bench_data_bundle or len(bench_name_list) < 3:
        return None
    layers = bench_name_list[2].split(Const.SEP)

    def _get(key, container):
        if isinstance(container, dict):
            return container.get(key)
        if isinstance(container, list):
            try:
                return container[int(key)]
            except (ValueError, IndexError):
                return None
        return None

    def get_by_layer(container, params_grad=False):
        data = container
        # dump.json中parameters_grad的结构为key：[{}], 如果存在key，有且只有一个列表元素，而op_name中只命名到了key，因此加'0'
        if params_grad:
            layers.append('0')
        for layer in layers:
            data = _get(layer, data)
        return _get(CompareConst.DATA_NAME.lower(), data)

    if Const.INPUT == bench_name_list[1]:
        return get_by_layer(bench_data_bundle.get(Const.INPUT, bench_data_bundle.get(Const.INPUT_ARGS)))
    elif Const.KWARGS == bench_name_list[1]:
        return get_by_layer(bench_data_bundle.get(Const.INPUT_KWARGS))
    elif Const.OUTPUT == bench_name_list[1]:
        return get_by_layer(bench_data_bundle.get(Const.OUTPUT))
    elif Const.PARAMS == bench_name_list[1]:
        return get_by_layer(bench_data_bundle.get(Const.PARAMS))
    elif Const.PARAMS_GRAD == bench_name_list[1]:
        return get_by_layer(bench_data_bundle, params_grad=True)
    else:
        return None
