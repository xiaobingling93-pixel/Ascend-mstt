# 进行比对及结果展示
import os
from rich.table import Table
from rich.console import Console
from api_accuracy_checker.compare.algorithm import compare_core
from api_accuracy_checker.common.utils import get_json_contents, write_csv
from api_accuracy_checker.compare.compare_utils import CompareConst
from api_accuracy_checker.common.config import msCheckerConfig


class Comparator:
    # consts for result csv
    COLUMN_API_NAME = "API name"
    COLUMN_FORWARD_SUCCESS = "Forward Test Success"
    COLUMN_BACKWARD_SUCCESS = "Backward Test Success"
    COLUMN_STACK_INFO = "Traceback callstack info"

    def __init__(self, result_csv_path, details_csv_path, is_continue_run_ut, test_result_cnt=None, stack_info_json_path=None):
        self.save_path = result_csv_path
        self.detail_save_path = details_csv_path
        if not is_continue_run_ut:
            if os.path.exists(self.save_path):
                raise ValueError(f"file {self.save_path} already exists, please remove it first or use a new dump path")
            if os.path.exists(self.detail_save_path):
                raise ValueError(
                    f"file {self.detail_save_path} already exists, please remove it first or use a new dump path")
            self.write_csv_title()
        if stack_info_json_path:
            self.stack_info = get_json_contents(stack_info_json_path)
        else:
            self.stack_info = None

        self.test_result_cnt = {
            "forward_fail_num": 0, "backward_fail_num": 0, "forward_and_backward_fail_num": 0, "success_num": 0,
            "total_num": 0, "forward_or_backward_fail_num": 0
        } if not test_result_cnt else test_result_cnt

    def print_pretest_result(self):
        if self.test_result_cnt.get("total_num") != 0:
            passing_rate = str(self.test_result_cnt.get("success_num") / self.test_result_cnt.get("total_num"))
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

    def write_csv_title(self):
        summary_test_rows = [[self.COLUMN_API_NAME, self.COLUMN_FORWARD_SUCCESS, self.COLUMN_BACKWARD_SUCCESS, "Message"]]
        write_csv(summary_test_rows, self.save_path)

        detail_test_rows = [[
            "Npu Name", "Bench Dtype", "NPU Dtype", "Shape",
            "Cosine Similarity",
            "Max Abs Error",
            "Relative Error (dual hundredth)",
            "Relative Error (dual thousandth)",
            "Relative Error (dual ten thousandth)",
            "Error Rate",
            "Status",
            "Message"
        ]]
        write_csv(detail_test_rows, self.detail_save_path)

    def write_summary_csv(self, test_result):
        test_rows = []
        if self.stack_info:
            test_rows[0].append(self.COLUMN_STACK_INFO)

        name = test_result[0]
        df_row = list(test_result[:3])
        if test_result[1] == "SKIP" or test_result[2] == "SKIP":
            df_row.append(test_result[3])
        if self.stack_info:
            stack_info = "\n".join(self.stack_info[name])
            df_row.append(stack_info)
        test_rows.append(df_row)
        write_csv(test_rows, self.save_path)

    def write_detail_csv(self, test_result):
        test_rows = []

        subject_prefix = test_result[0]
        fwd_result = test_result[3]
        bwd_result = test_result[4]
        if isinstance(fwd_result, list):
            for i, test_subject in enumerate(fwd_result):
                subject = subject_prefix + ".forward.output." + str(i)
                test_subject = ["{:.{}f}".format(item, msCheckerConfig.precision) if isinstance(item, float) else item for item in test_subject]
                test_rows.append([subject] + list(test_subject))
        if isinstance(bwd_result, list):
            for i, test_subject in enumerate(bwd_result):
                subject = subject_prefix + ".backward.output." + str(i)
                test_subject = ["{:.{}f}".format(item, msCheckerConfig.precision) if isinstance(item, float) else item for item in test_subject]
                test_rows.append([subject] + list(test_subject))

        write_csv(test_rows, self.detail_save_path)

    def record_results(self, *args):
        self.write_summary_csv(args)
        self.write_detail_csv(args)

    def compare_output(self, api_name, bench_out, npu_out, bench_grad=None, npu_grad=None):
        self.test_result_cnt["total_num"] += 1
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
            self.record_results(api_name, is_fwd_success, CompareConst.NA, fwd_compare_alg_results,
                                bwd_compare_alg_results)
        else:
            self.record_results(api_name, is_fwd_success, is_bwd_success, fwd_compare_alg_results,
                                bwd_compare_alg_results)
        if is_fwd_success and is_bwd_success:
            self.test_result_cnt['success_num'] += 1
        elif not is_fwd_success and not is_bwd_success:
            self.test_result_cnt['forward_and_backward_fail_num'] += 1
        elif not is_fwd_success:
            self.test_result_cnt['forward_fail_num'] += 1
            self.test_result_cnt['forward_or_backward_fail_num'] += 1
        else:
            self.test_result_cnt['backward_fail_num'] += 1
            self.test_result_cnt['forward_or_backward_fail_num'] += 1
        return is_fwd_success, is_bwd_success

    @staticmethod
    def _compare_core_wrapper(bench_out, npu_out):
        detailed_result_total = []
        test_final_success = True
        status, compare_result, message = compare_core(bench_out, npu_out)
        if not isinstance(status, list):
            detailed_result_total.append(compare_result.to_column_value(status, message))
            if status in [CompareConst.ERROR, CompareConst.WARNING]:
                test_final_success = False
        else:
            for item, item_status in enumerate(status):
                detailed_result_total.append(compare_result[item].to_column_value(item_status, message[item]))
                if item_status in [CompareConst.ERROR, CompareConst.WARNING]:
                    test_final_success = False
        return test_final_success, detailed_result_total

    @staticmethod
    def _compare_dropout(bench_out, npu_out):
        tensor_num = bench_out.numel()
        if tensor_num >= 100:
            if abs((bench_out == 0).sum() - (npu_out == 0).cpu().sum()) / tensor_num < 0.1:
                return True, 1
            else:
                return False, 0
        else:
            return True, 1
