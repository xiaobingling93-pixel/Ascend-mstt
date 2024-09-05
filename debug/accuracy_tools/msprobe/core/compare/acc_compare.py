import multiprocessing
import os
import json
import pandas as pd
from msprobe.core.common.file_utils import FileOpen
from msprobe.core.common.const import CompareConst, Const
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.log import logger
from msprobe.core.common.utils import add_time_with_xlsx, CompareException
from msprobe.core.common.file_utils import remove_path
from msprobe.core.compare.check import check_graph_mode, check_struct_match, fuzzy_check_op
from msprobe.core.compare.highlight import find_compare_result_error_rows, highlight_rows_xlsx
from msprobe.core.compare.utils import read_op, merge_tensor, get_un_match_accuracy, get_accuracy
from msprobe.core.compare.multiprocessing_compute import _handle_multi_process, ComparisonResult, _save_cmp_result
from msprobe.core.compare.npy_compare import compare_ops_apply, get_error_type, reshape_value, get_relative_err, \
    get_error_message
from msprobe.core.advisor.advisor import Advisor


class Comparator:
    
    def __init__(self):
        pass
    
    @classmethod
    def make_result_table(cls,result, md5_compare, summary_compare, stack_mode):
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
    
    @classmethod
    def gen_merge_list(self, json_data, op_name,stack_json_data, summary_compare, md5_compare):
        op_data = json_data['data'][op_name]
        op_parsed_list = read_op(op_data, op_name)
        if op_name in stack_json_data:
            op_parsed_list.append({'full_op_name': op_name, 'full_info': stack_json_data[op_name]})
        else:
            op_parsed_list.append({'full_op_name': op_name, 'full_info': None})
            
        merge_list = merge_tensor(op_parsed_list, summary_compare, md5_compare)
        return merge_list
    
    def check_op(self, npu_dict, bench_dict, fuzzy_match):
        a_op_name = npu_dict["op_name"]
        b_op_name = bench_dict["op_name"]
        graph_mode = check_graph_mode(a_op_name[0], b_op_name[0])
        
        frame_name = getattr(self,"frame_name")
        if frame_name == "PTComparator":
            from msprobe.pytorch.compare.match import graph_mapping
            if graph_mode:
                return graph_mapping.match(a_op_name[0], b_op_name[0])
        struct_match = check_struct_match(npu_dict, bench_dict)
        if not fuzzy_match:
            return a_op_name == b_op_name and struct_match
        is_match = True
        try:
            is_match = fuzzy_check_op(a_op_name, b_op_name)
        except Exception as err:
            logger.warning("%s and %s can not fuzzy match." % (a_op_name, b_op_name))
            is_match = False
        return is_match and struct_match
    
    def match_op(self, npu_queue, bench_queue, fuzzy_match):
        for b_index, b_op in enumerate(bench_queue[0: -1]):
            if self.check_op(npu_queue[-1], b_op, fuzzy_match):
                return len(npu_queue) - 1, b_index
        if self.check_op(npu_queue[-1], bench_queue[-1], fuzzy_match):
            return len(npu_queue) - 1, len(bench_queue) - 1
        for n_index, n_op in enumerate(npu_queue[0: -1]):
            if self.check_op(n_op, bench_queue[-1], fuzzy_match):
                return n_index, len(bench_queue) - 1
        return -1, -1
    
    def compare_process(self, file_handles, stack_mode, fuzzy_match, summary_compare=False, md5_compare=False):
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
                npu_merge_list = self.gen_merge_list(npu_json_data,op_name_npu,stack_json_data,summary_compare,md5_compare)
                if npu_merge_list:
                    npu_ops_queue.append(npu_merge_list)
            except StopIteration:
                read_err_npu = False
            try:
                last_bench_ops_len = len(bench_ops_queue)
                op_name_bench = next(ops_bench_iter)
                bench_merge_list = self.gen_merge_list(bench_json_data,op_name_bench,stack_json_data,summary_compare,md5_compare)
                if bench_merge_list:
                    bench_ops_queue.append(bench_merge_list)
            except StopIteration:
                read_err_bench = False

            # merge all boolean expressions
            both_empty = not npu_ops_queue and not bench_ops_queue
            no_change = (len(npu_ops_queue) == last_npu_ops_len) and (len(bench_ops_queue) == last_bench_ops_len)
            if both_empty or no_change:
                continue

            # APIs in NPU and Bench models unconsistent judgment
            if bool(npu_ops_queue) ^ bool(bench_ops_queue):
                logger.info("Please check whether the number and calls of APIs in NPU and Bench models are consistent.")
                break

            n_match_point, b_match_point = self.match_op(npu_ops_queue, bench_ops_queue, fuzzy_match)
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
                
        result_df = self.make_result_table(result, md5_compare, summary_compare, stack_mode)
        return result_df
    
    def compare_by_op(self, npu_op_name, bench_op_name, op_name_mapping_dict, input_param):
        npu_bench_name_list = op_name_mapping_dict[npu_op_name]
        data_name = npu_bench_name_list[1]
        error_file, relative_err, error_flag = None, None, False
        if data_name == '-1' or data_name == -1:  # 没有真实数据路径
            n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
            error_flag = True
        else:
            try:
                read_npy_data = getattr(self, "read_npy_data")
                frame_name = getattr(self, "frame_name")
                if frame_name == "MSComparator":
                    n_value = read_npy_data(input_param.get("npu_dump_data_dir"), npu_op_name + Const.NUMPY_SUFFIX)
                    if self.cross_frame:
                        b_value = read_npy_data(input_param.get("bench_dump_data_dir"), bench_op_name + Const.PT_SUFFIX, load_pt_file=True)
                    else:
                        b_value = read_npy_data(input_param.get("bench_dump_data_dir"), bench_op_name + Const.NUMPY_SUFFIX)
                else:
                    n_value = read_npy_data(input_param.get("npu_dump_data_dir"), npu_op_name + Const.PT_SUFFIX)
                    b_value = read_npy_data(input_param.get("bench_dump_data_dir"), bench_op_name + Const.PT_SUFFIX)
            except IOError as error:
                error_file = error.filename
                n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
                error_flag = True
            except FileCheckException:
                error_file = data_name
                n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
                error_flag = True

        n_value, b_value, error_flag = get_error_type(n_value, b_value, error_flag)
        if not error_flag:
            relative_err = get_relative_err(n_value, b_value)
            n_value, b_value = reshape_value(n_value, b_value)

        err_msg = get_error_message(n_value, b_value, npu_op_name, error_flag, error_file=error_file)
        result_list, err_msg = compare_ops_apply(n_value, b_value, error_flag, err_msg, relative_err=relative_err)

        if npu_op_name != bench_op_name and bench_op_name != CompareConst.N_A:
            err_msg += " Fuzzy matching data, the comparison accuracy may be affected."
        result_list.append(err_msg)
        return result_list
    
    def compare_core(self, input_parma, output_path, **kwargs):
        """
        Compares data from multiple JSON files and generates a comparison report.

        Args:
            input_parma (dict): A dictionary containing paths to JSON files ("npu_path", "bench_path",
                                "stack_path").
            output_path (str): The path where the output Excel report will be saved.
            **kwargs: Additional keyword arguments including:
            - stack_mode (bool, optional): Enables stack mode comparison. Defaults to False.
            - auto_analyze (bool, optional): If True, triggers automatic analysis after comparison. Defaults to True.
            - suffix (str, optional): Suffix to append to the output file name. Defaults to ''.
            - fuzzy_match (bool, optional): Enables fuzzy matching during comparison. Defaults to False.
            - summary_compare (bool, optional): Enables summary comparison mode. Defaults to False.
            - md5_compare (bool, optional): Enables MD5 comparison. Defaults to False.

        Returns:
        """
        # get kwargs or set default value
        stack_mode = kwargs.get('stack_mode', False)
        auto_analyze = kwargs.get('auto_analyze', True)
        suffix = kwargs.get('suffix', '')
        fuzzy_match = kwargs.get('fuzzy_match', False)
        summary_compare = kwargs.get('summary_compare', False)
        md5_compare = kwargs.get('md5_compare', False)

        logger.info("Please check whether the input data belongs to you. If not, there may be security risks.")
        file_name = add_time_with_xlsx("compare_result" + suffix)
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        remove_path(file_path)
        highlight_dict = {'red_rows': [], 'yellow_rows': []}
        
        with FileOpen(input_parma.get("npu_json_path"), "r") as npu_json, \
                FileOpen(input_parma.get("bench_json_path"), "r") as bench_json, \
                FileOpen(input_parma.get("stack_json_path"), "r") as stack_json:
            result_df = self.compare_process([npu_json, bench_json, stack_json], stack_mode, fuzzy_match,
                                        summary_compare, md5_compare)

        if not md5_compare and not summary_compare:
            result_df = self._do_multi_process(input_parma, result_df)
        find_compare_result_error_rows(result_df, highlight_dict, summary_compare, md5_compare)
        highlight_rows_xlsx(result_df, highlight_dict, file_path)
        if auto_analyze:
            advisor = Advisor(result_df, output_path)
            advisor.analysis()
    
    def compare_ops(self, idx, dump_path_dict, result_df, lock, input_param):
        cos_result = []
        max_err_result = []
        max_relative_err_result = []
        err_mess = []
        one_thousand_err_ratio_result = []
        five_thousand_err_ratio_result = []
        is_print_compare_log = input_param.get("is_print_compare_log")
        for i in range(len(result_df)):
            npu_op_name = result_df.iloc[i, 0]
            bench_op_name = result_df.iloc[i, 1]
            if is_print_compare_log:
                logger.info("start compare: {}".format(npu_op_name))
            cos_sim, max_abs_err, max_relative_err, one_thousand_err_ratio, five_thousand_err_ratio, err_msg = self.compare_by_op(
                npu_op_name, bench_op_name, dump_path_dict, input_param)
            if is_print_compare_log:
                logger.info(
                    "[{}] Compare result: cosine {}, max_abs_err {}, max_relative_err {}, {}, one_thousand_err_ratio {}, "
                    "five_thousand_err_ratio {}".format(npu_op_name, cos_sim, max_abs_err, max_relative_err, err_msg,
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
    
    def _do_multi_process(self,input_parma, result_df):
        try:
            result_df = _handle_multi_process(self.compare_ops, input_parma, result_df, multiprocessing.Manager().RLock())
            return result_df
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e
    