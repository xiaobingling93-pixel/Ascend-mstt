from msprobe.core.compare.check import check_op
from msprobe.core.common.const import CompareConst
from msprobe.core.compare.npy_compare import compare_ops_apply, get_error_type, reshape_value, get_relative_err, \
    get_error_message
from msprobe.core.common.exceptions import FileCheckException


class Comparator:
    
    def __init__(self):
        pass    
    
    @classmethod
    def match_op(cls,npu_queue, bench_queue, fuzzy_match):
        for b_index, b_op in enumerate(bench_queue[0: -1]):
            if check_op(npu_queue[-1], b_op, fuzzy_match):
                return len(npu_queue) - 1, b_index
        if check_op(npu_queue[-1], bench_queue[-1], fuzzy_match):
            return len(npu_queue) - 1, len(bench_queue) - 1
        for n_index, n_op in enumerate(npu_queue[0: -1]):
            if check_op(n_op, bench_queue[-1], fuzzy_match):
                return n_index, len(bench_queue) - 1
        return -1, -1
    
    def compare_by_op(self,op_name, op_name_mapping_dict, input_parma):
        npu_bench_name_list = op_name_mapping_dict[op_name]
        data_name = npu_bench_name_list[1]
        error_file, relative_err, error_flag = None, None, False
        if data_name == '-1' or data_name == -1:  # 没有真实数据路径
            n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
            error_flag = True
        else:
            try:
                read_npy_data=getattr(self,"read_npy_data")
                n_value = read_npy_data(input_parma.get("npu_dump_data_dir"), npu_bench_name_list[0])
                b_value = read_npy_data(input_parma.get("bench_dump_data_dir"), npu_bench_name_list[1])
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

        err_msg = get_error_message(n_value, b_value, op_name, error_flag, error_file=error_file)
        result_list, err_msg = compare_ops_apply(n_value, b_value, error_flag, err_msg, relative_err=relative_err)

        if npu_bench_name_list[0] != npu_bench_name_list[1]:
            err_msg += " Fuzzy matching data, the comparison accuracy may be affected."
        result_list.append(err_msg)
        return result_list
    
