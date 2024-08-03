from msprobe.core.compare.acc_compare import Comparator 
from msprobe.core.common.log import logger






import json
import multiprocessing
import os.path
import sys

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill
from collections import namedtuple
from dataclasses import dataclass

from msprobe.mindspore.compare.match import graph_mapping
from msprobe.mindspore.compare.highlight import HighlightRules, get_header_index

from msprobe.mindspore.advisor.advisor import Advisor
from msprobe.mindspore.common.log import logger
from msprobe.core.common.utils import check_compare_param, add_time_with_xlsx, CompareException, \
    format_value, check_file_not_exists, check_configuration_param, task_dumppath_get
from msprobe.core.common.file_check import FileChecker, change_mode, FileOpen, create_directory
from msprobe.core.common.const import Const, CompareConst, FileCheckConst
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.compare.utils import ComparisonResult,_save_cmp_result,merge_tensor, get_un_match_accuracy,get_accuracy,read_op


class MSComparator (Comparator):
    def __init__(self):
        super().__init__()
    
    
    def compare_ops(self,idx, dump_path_dict, result_df, lock, input_parma):
        cos_result = []
        max_err_result = []
        max_relative_err_result = []
        err_mess = []
        one_thousand_err_ratio_result = []
        five_thousand_err_ratio_result = []
        is_print_compare_log = input_parma.get("is_print_compare_log")
        for i in range(len(result_df)):
            op_name = result_df.iloc[i, 0]
            if is_print_compare_log:
                logger.info("start compare: {}".format(op_name))
            cos_sim, max_abs_err, max_relative_err, one_thousand_err_ratio, five_thousand_err_ratio, err_msg = self.compare_by_op(
                op_name, dump_path_dict, input_parma)
            if is_print_compare_log:
                logger.info(
                    "[{}] Compare result: cosine {}, max_abs_err {}, max_relative_err {}, {}, one_thousand_err_ratio {}, "
                    "five_thousand_err_ratio {}".format(op_name, cos_sim, max_abs_err, max_relative_err, err_msg,
                                                        one_thousand_err_ratio, five_thousand_err_ratio))
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
    
    def compare_process(self,file_handles, stack_mode, fuzzy_match, summary_compare=False, md5_compare=False):
        npu_json_handle, bench_json_handle, stack_json_handle = file_handles
        npu_json_data = json.load(npu_json_handle)
        bench_json_data = json.load(bench_json_handle)
        stack_json_data = json.load(stack_json_handle)

        if fuzzy_match:
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

        while True:
            if not read_err_npu and not read_err_bench:
                break
            try:
                last_npu_ops_len = len(npu_ops_queue)
                op_name_npu = next(ops_npu_iter)
                read_err_npu = True

                npu_op_data = npu_json_data['data'][op_name_npu]
                npu_op_parsed_list = read_op(npu_op_data, op_name_npu)
                if op_name_npu in stack_json_data:
                    npu_op_parsed_list.append({'full_op_name': op_name_npu, 'full_info': stack_json_data[op_name_npu]})
                else:
                    npu_op_parsed_list.append({'full_op_name': op_name_npu, 'full_info': None})

                npu_merge_list = merge_tensor(npu_op_parsed_list, summary_compare, md5_compare)
                if npu_merge_list:
                    npu_ops_queue.append(npu_merge_list)
            except StopIteration:
                read_err_npu = False
            try:
                last_bench_ops_len = len(bench_ops_queue)
                op_name_bench = next(ops_bench_iter)

                bench_op_data = bench_json_data['data'][op_name_bench]
                bench_op_parsed_list = read_op(bench_op_data, op_name_bench)
                if op_name_bench in stack_json_data:
                    bench_op_parsed_list.append(
                        {'full_op_name': op_name_bench, 'full_info': stack_json_data[op_name_bench]})
                else:
                    bench_op_parsed_list.append({'full_op_name': op_name_bench, 'full_info': None})

                bench_merge_list = merge_tensor(bench_op_parsed_list, summary_compare, md5_compare)
                if bench_merge_list:
                    bench_ops_queue.append(bench_merge_list)
            except StopIteration:
                read_err_bench = False

            # merge all boolean expressions
            both_empty = not npu_ops_queue and not bench_ops_queue
            no_change = (len(npu_ops_queue) == last_npu_ops_len) and (len(bench_ops_queue) == last_bench_ops_len)
            if both_empty or no_change:
                continue

            n_match_point, b_match_point =  super().match_op(npu_ops_queue, bench_ops_queue, fuzzy_match)
            if n_match_point == -1 and b_match_point == -1:
                continue
            n_match_data = npu_ops_queue[n_match_point]
            b_match_data = bench_ops_queue[b_match_point]
            un_match_data = npu_ops_queue[0: n_match_point]
            for npu_data in un_match_data:
                get_un_match_accuracy(result, npu_data, md5_compare, summary_compare)
            get_accuracy(result, n_match_data, b_match_data, summary_compare, md5_compare)
            del npu_ops_queue[0: n_match_point + 1]
            del bench_ops_queue[0: b_match_point + 1]
        if npu_ops_queue:
            for npu_data in npu_ops_queue:
                get_un_match_accuracy(result, npu_data, md5_compare, summary_compare)

        header = []
        if md5_compare:
            header = CompareConst.MD5_COMPARE_RESULT_HEADER[:]
        elif summary_compare:
            header = CompareConst.SUMMARY_COMPARE_RESULT_HEADER[:]
        else:
            header = CompareConst.COMPARE_RESULT_HEADER[:]

        all_mode_bool = not (summary_compare or md5_compare)
        if stack_mode:
            if all_mode_bool:
                header.append(CompareConst.STACK)
                header.append(CompareConst.DATA_NAME)
            else:
                header.append(CompareConst.STACK)
        else:
            if all_mode_bool:
                for row in result:
                    del row[-2]
                header.append(CompareConst.DATA_NAME)
            else:
                for row in result:
                    del row[-1]

        result_df = pd.DataFrame(result, columns=header)
        return result_df
    
    
    def read_npy_data(self,dir_path, file_name):
        data_path = os.path.join(dir_path, file_name)
        path_checker = FileChecker(data_path, FileCheckConst.FILE, FileCheckConst.READ_ABLE,
                                FileCheckConst.NUMPY_SUFFIX, False)
        data_path = path_checker.common_check()
        data_value = np.load(data_path)      # detach for less memory
        if data_value.dtype == np.float16:
            data_value=data_value.astype(np.float32)

        return data_value
    
    
    

    
    
    
        
    