import multiprocessing
import pandas as pd
from msprobe.core.common.const import CompareConst, Const
from msprobe.core.compare.npy_compare import compare_ops_apply, get_error_type, reshape_value, get_relative_err, \
    get_error_message
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.compare.utils import read_op, merge_tensor, CompareException
from msprobe.core.compare.multiprocessing_compute import _handle_multi_process
from msprobe.core.common.log import logger
from msprobe.core.compare.check import check_graph_mode, check_struct_match, fuzzy_check_op


class Comparator:
    
    def __init__(self):
        pass    
    
    @classmethod
    def make_result_table(cls,result,md5_compare,summary_compare,stack_mode):
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
    def gen_merge_list(self,json_data,op_name,stack_json_data,summary_compare,md5_compare):
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

        if npu_op_name != bench_op_name:
            err_msg += " Fuzzy matching data, the comparison accuracy may be affected."
        result_list.append(err_msg)
        return result_list
    
    def _do_multi_process(self,input_parma, result_df):
        try:
            compare_ops = getattr(self,"compare_ops")
            result_df = _handle_multi_process(compare_ops, input_parma, result_df, multiprocessing.Manager().RLock())
            return result_df
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e
    