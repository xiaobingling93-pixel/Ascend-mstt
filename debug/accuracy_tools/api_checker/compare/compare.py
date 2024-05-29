import os
from datetime import datetime, timezone
from compare.algorithm import CompareColumn, compare_float_tensor, compare_bool_tensor
from common.utils import get_json_contents, write_csv, np_scalar_type


class Comparator:
    # consts for result csv
    RESULT_CSV_PATH = "_result.csv"
    DETAILS_CSV_PATH = "_detail.csv"
    
    def __init__(self, outpath, is_continue_run_ut, stack_info_json_path=None):
        time = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
        self.save_path = os.path.join(outpath, time + self.RESULT_CSV_PATH)
        self.detail_save_path = os.path.join(outpath, time + self.DETAILS_CSV_PATH)
        if not is_continue_run_ut and not os.path.exists(self.save_path) and not os.path.exists(self.detail_save_path):
            self.write_csv_title()
        if stack_info_json_path:
            self.stack_info = get_json_contents(stack_info_json_path)
        else:
            self.stack_info = None

        self.test_result_cnt = {
            "forward_fail_num": 0, "backward_fail_num": 0, "forward_and_backward_fail_num": 0, "success_num": 0,
            "total_num": 0, "forward_or_backward_fail_num": 0
        }
    
    def compare(self, bench_out, npu_out, api_name): 
        compareColumn = CompareColumn()
        compareColumn.bench_type = bench_out.dtype
        compareColumn.npu_type = npu_out.dtype
        compareColumn.shape = npu_out.shape
        if npu_out.dtype in np_scalar_type:
             err_rate, status, message = compare_bool_tensor(bench_out, npu_out)
             compareColumn.err_rate = err_rate
        else:
             status, compareColumn, message = compare_float_tensor(bench_out, npu_out, compareColumn)
        result_list = [api_name, status]
        write_csv(result_list, self.save_path)
        detail_list = [api_name]
        detail_temp = compareColumn.to_column_value(status, message)
        for detail in detail_temp:
             detail_list.append(detail)
        write_csv(detail_list, self.detail_save_path)

    def write_result_csv(self, detail_dict):
            result_list = [detail_dict["api_name"], detail_dict["status"]]
            write_csv(result_list, self.save_path)
                
    def write_csv_title(self):
        result_test_rows = [
            "API name",
            "Forward Test Success",
            "Backward Test Success",
            "Message"
        ]
        write_csv(result_test_rows, self.save_path)

        detail_test_rows = [
            "API Name", "Bench Dtype", "NPU Dtype", "Shape",
            "余弦相似度",
            "最大绝对误差",
            "双百指标",
            "双千指标",
            "双万指标",
            "错误率",
            "误差均衡性",
            "均方根误差",
            "小值域错误占比",
            "相对误差最大值",
            "相对误差平均值",
            "Status",
            "Message"
        ]
        write_csv(detail_test_rows, self.detail_save_path)

