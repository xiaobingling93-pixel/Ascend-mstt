# 进行比对及结果展示
import os
import sys
import csv
import json
from collections import namedtuple
from rich.table import Table
from rich.console import Console
from msprobe.core.common.const import CompareConst, FileCheckConst
from msprobe.core.common.file_utils import FileOpen, change_mode, read_csv
from msprobe.pytorch.online_dispatch.single_compare import single_benchmark_compare_wrap
from msprobe.pytorch.common.log import logger
from msprobe.core.common.utils import CompareException, check_op_str_pattern_valid

ELEMENT_NUM_THRESHOLD = 100
ZERO_NUM_THRESHOLD = 0.1
FLOAT_PRECISION = 14

ResultInfo = namedtuple('ResultInfo', ['api_name', 'is_fwd_success', 'is_bwd_success',
                                       'fwd_compare_alg_results', 'bwd_compare_alg_results'])

def get_file_content_bytes(file):
    with FileOpen(file, 'rb') as file_handle:
        return file_handle.read()


def get_json_contents(file_path):
    ops = get_file_content_bytes(file_path)
    try:
        json_obj = json.loads(ops)
    except ValueError as error:
        logger.error('Failed to load "%s". %s' % (file_path, str(error)))
        raise CompareException(CompareException.INVALID_FILE_ERROR) from error
    if not isinstance(json_obj, dict):
        logger.error('Json file %s, content is not a dictionary!' % file_path)
        raise CompareException(CompareException.INVALID_FILE_ERROR)
    return json_obj


def write_csv(data, filepath):
    with FileOpen(filepath, 'a', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    change_mode(filepath, FileCheckConst.DATA_FILE_AUTHORITY)


class Saver:
    # consts for result csv
    COLUMN_API_NAME = "API name"
    COLUMN_FORWARD_SUCCESS = "Forward Test Success"
    COLUMN_BACKWARD_SUCCESS = "Backward Test Success"
    COLUMN_STACK_INFO = "Traceback callstack info"

    def __init__(self, save_path, detail_save_path, stack_info):
        self.save_path = save_path
        self.detail_save_path = detail_save_path
        self.stack_info = stack_info

        self.test_result_cnt = {
            "forward_fail_num": 0, "backward_fail_num": 0, "forward_and_backward_fail_num": 0, "success_num": 0,
            "total_num": 0, "forward_or_backward_fail_num": 0
        }

    def write_csv_title(self):
        summary_test_rows = [[self.COLUMN_API_NAME, self.COLUMN_FORWARD_SUCCESS, self.COLUMN_BACKWARD_SUCCESS, "Message"]]
        write_csv(summary_test_rows, self.save_path)

        detail_test_rows = [[
            "Npu Name", "Bench Dtype", "NPU Dtype", "Shape",
            "error_balance", "max_abs_diff", "max_abs_idx",
            "max_rel_diff", "max_rel_idx", "eb_thd",
            "error_thd", "Status","Message"
        ]]
        write_csv(detail_test_rows, self.detail_save_path)

    def print_pretest_result(self):
        self.get_statistics_from_result_csv()
        if self.test_result_cnt.get("total_num") != 0:
            passing_rate = str(self.test_result_cnt.get("success_num") /
                               (self.test_result_cnt.get("total_num") + sys.float_info.epsilon))
        else:
            passing_rate = "0"

        console = Console()
        table_total = Table(
            show_header=True, title="Overall Statistics", show_lines=True, width=75
        )
        table_total.add_column("Result")
        table_total.add_column("Statistics")
        table_total.add_row("[green]Pass[/green]", str(self.test_result_cnt.get("success_num")))
        table_total.add_row("[red]Fail[/red]", str(self.test_result_cnt.get("forward_and_backward_fail_num") +
                                                   self.test_result_cnt.get("forward_or_backward_fail_num")))
        table_total.add_row("Passing Rate", passing_rate)

        table_detail = Table(
            show_header=True, title="Detail Statistics", show_lines=True, width=75
        )
        table_detail.add_column("Result")
        table_detail.add_column("Statistics")
        table_detail.add_row("Only Forward Fail", str(self.test_result_cnt.get("forward_fail_num")))
        table_detail.add_row("Only Backward Fail", str(self.test_result_cnt.get("backward_fail_num")))
        table_detail.add_row(
            "Both Forward & Backward Fail", str(self.test_result_cnt.get("forward_and_backward_fail_num")))

        console.print(table_total)
        console.print(table_detail)

    def get_statistics_from_result_csv(self):
        checklist = [CompareConst.TRUE, CompareConst.FALSE, CompareConst.NA, CompareConst.SKIP]
        data = read_csv(self.save_path)
        result_csv_name = os.path.basename(self.save_path)
        for _, row in data.iterrows():
            if len(row) < 3:
                raise ValueError("The number of columns in %s is incorrect" % result_csv_name)
            if not all(row[i] and row[i].upper() in checklist for i in (1, 2)):
                raise ValueError(
                    "The value in the 2nd or 3rd column of %s is wrong, it must be TRUE, FALSE, SKIP or N/A"
                    % result_csv_name)
            column1 = row[1].upper()
            column2 = row[2].upper()
            if column1 == CompareConst.SKIP:
                continue
            self.test_result_cnt["total_num"] += 1
            if column1 == CompareConst.TRUE and column2 in [CompareConst.TRUE, 'N/A']:
                self.test_result_cnt['success_num'] += 1
            elif column1 == CompareConst.FALSE and column2 == CompareConst.FALSE:
                self.test_result_cnt['forward_and_backward_fail_num'] += 1
            elif column1 == CompareConst.FALSE:
                self.test_result_cnt['forward_fail_num'] += 1
                self.test_result_cnt['forward_or_backward_fail_num'] += 1
            else:
                self.test_result_cnt['backward_fail_num'] += 1
                self.test_result_cnt['forward_or_backward_fail_num'] += 1

    def write_summary_csv(self, test_result):
        test_rows = []
        if self.stack_info:
            test_rows[0].append(self.COLUMN_STACK_INFO)

        check_op_str_pattern_valid(test_result.api_name)
        df_row = [test_result.api_name, test_result.is_fwd_success, test_result.is_bwd_success]
        if test_result.is_fwd_success == "SKIP" or test_result.is_bwd_success == "SKIP":
            df_row.append(test_result.fwd_compare_alg_results)
        if self.stack_info:
            check_op_str_pattern_valid(self.stack_info[test_result.api_name])
            stack_info = "\n".join(self.stack_info[test_result.api_name])
            df_row.append(stack_info)
        test_rows.append(df_row)
        write_csv(test_rows, self.save_path)

    def write_detail_csv(self, test_result):
        def get_rows_from_list(result, name, sub_prefix):
            rows = []
            if isinstance(result, list):
                for i, test_subject in enumerate(result):
                    subject = sub_prefix + "." + name + ".output." + str(i)
                    test_subject = ["{:.{}f}".format(item, FLOAT_PRECISION) if isinstance(item, float) else item for
                                    item in test_subject]
                    rows.append([subject] + list(test_subject))
            return rows

        test_rows = []
        subject_prefix = test_result.api_name
        fwd_result = test_result.fwd_compare_alg_results
        bwd_result = test_result.bwd_compare_alg_results

        test_rows.extend(get_rows_from_list(fwd_result, "forward", subject_prefix))
        test_rows.extend(get_rows_from_list(bwd_result, "backward", subject_prefix))

        write_csv(test_rows, self.detail_save_path)

    def record_results(self, result_info):
        self.write_summary_csv(result_info)
        self.write_detail_csv(result_info)


class Comparator:

    def __init__(self, result_csv_path, details_csv_path, is_continue_run_ut, stack_info_json_path=None):
        self.save_path = result_csv_path
        self.detail_save_path = details_csv_path
        if stack_info_json_path:
            self.stack_info = get_json_contents(stack_info_json_path)
        else:
            self.stack_info = None
        self.saver = Saver(result_csv_path, details_csv_path, self.stack_info)

        if is_continue_run_ut and not os.path.exists(self.save_path) and not os.path.exists(self.detail_save_path):
            self.saver.write_csv_title()

    @staticmethod
    def _compare_core_wrapper(bench_out, npu_out):
        detailed_result_total = []
        test_final_success = True
        status, details = single_benchmark_compare_wrap(npu_out, bench_out)
        if not isinstance(status, list):
            detailed_result_total.append(details)
            test_final_success = status
        else:
            for item, item_status in enumerate(status):
                detailed_result_total.append(details.get(item, 'key does not exist'))
                if not item_status:
                    test_final_success = False
        return test_final_success, detailed_result_total

    @staticmethod
    def _compare_dropout(bench_out, npu_out):
        tensor_num = bench_out.numel()
        if tensor_num >= ELEMENT_NUM_THRESHOLD:
            if abs((bench_out == 0).sum() - (npu_out == 0).cpu().sum()) / tensor_num < ZERO_NUM_THRESHOLD:
                return True, 1
            else:
                return False, 0
        else:
            return True, 1

    def compare_output(self, api_name, bench_out, npu_out, bench_grad=None, npu_grad=None):
        if "dropout" in api_name:
            is_fwd_success, fwd_compare_alg_results = self._compare_dropout(bench_out, npu_out)
        else:
            is_fwd_success, fwd_compare_alg_results = self._compare_core_wrapper(bench_out, npu_out)
        if bench_grad and npu_grad:
            if "dropout" in api_name:
                is_bwd_success, bwd_compare_alg_results = self._compare_dropout(bench_grad[0], npu_grad[0])
            else:
                is_bwd_success, bwd_compare_alg_results = self._compare_core_wrapper(bench_grad, npu_grad)
        else:
            is_bwd_success, bwd_compare_alg_results = True, None
        if is_bwd_success and bwd_compare_alg_results is None:
            self.saver.record_results(ResultInfo(api_name, is_fwd_success, CompareConst.NAN, fwd_compare_alg_results,
                                      bwd_compare_alg_results))
        else:
            self.saver.record_results(ResultInfo(api_name, is_fwd_success, is_bwd_success, fwd_compare_alg_results,
                                      bwd_compare_alg_results))
        return is_fwd_success, is_bwd_success
