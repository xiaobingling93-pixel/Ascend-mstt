import argparse
import os
import sys
import csv
import math
from collections import namedtuple
import pandas as pd

from api_accuracy_checker.common.utils import print_info_log, print_warn_log, print_error_log, write_csv, \
    CompareException, create_directory
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.compare.compare_utils import CompareConst, BENCHMARK_COMPARE_RESULT_FILE_NAME, \
BENCHMARK_COMPARE_DETAILS_FILE_NAME, Benchmark_Compare_Support_List, Api_Precision_Compare_Unsupport_List, \
    ApiPrecisionCompareColumn, AbsoluteStandardApiName, Binary_Compare_Unsupport_List
from api_accuracy_checker.run_ut.run_ut import get_validated_result_csv_path
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileCheckConst, FileChecker, change_mode
from ptdbg_ascend.src.python.ptdbg_ascend.common.utils import check_path_before_create


CompareConfig = namedtuple('CompareConfig', ['npu_csv_path', 'gpu_csv_path', 'result_csv_path', 'details_csv_path'])
unsupported_message = 'This data type does not support benchmark compare.'


benchmark_algorithms_thresholds = {
    'small_value' : {
        'error_threshold' : 2,
        'warning_threshold' : 1
    },
    'rmse' : {
        'error_threshold' : 2,
        'warning_threshold' : 1
    },
    'max_rel_err' : {
        'error_threshold' : 10,
        'warning_threshold' : 1
    },
    'mean_rel_err' : {
        'error_threshold' : 2,
        'warning_threshold' : 1
    },
    'eb' : {
        'error_threshold' : 2,
        'warning_threshold' : 1
    }
}


class BenchmarkStandard:
    def __init__(self, api_name, npu_precision, gpu_precision):
        self.api_name = api_name
        self.npu_precision = npu_precision
        self.gpu_precision = gpu_precision
        self.small_value_err_ratio = 1
        self.rmse_ratio = 1
        self.max_rel_err_ratio = 1
        self.mean_rel_err_ratio = 1
        self.eb_ratio = 1
        self.small_value_err_status = CompareConst.PASS
        self.rmse_status = CompareConst.PASS
        self.max_rel_err_status = CompareConst.PASS
        self.mean_rel_err_status = CompareConst.PASS
        self.eb_status = CompareConst.PASS
        self.check_result_list = []
        self.final_result = CompareConst.PASS

    def __str__(self):
        return "%s" % (self.api_name)

    def get_result(self):
        self._compare_ratio()
        self.small_value_err_status = self._get_status(self.small_value_err_ratio, 'small_value')
        self.check_result_list.append(self.small_value_err_status)
        self.rmse_status = self._get_status(self.rmse_ratio, 'rmse')
        self.check_result_list.append(self.rmse_status)
        self.max_rel_err_status = self._get_status(self.max_rel_err_ratio, 'max_rel_err')
        self.check_result_list.append(self.max_rel_err_status)
        self.mean_rel_err_status = self._get_status(self.mean_rel_err_ratio, 'mean_rel_err')
        self.check_result_list.append(self.mean_rel_err_status)
        self.eb_status = self._get_status(self.eb_ratio, 'eb')
        if CompareConst.ERROR in self.check_result_list:
            self.final_result = CompareConst.ERROR
        elif CompareConst.WARNING in self.check_result_list:
            self.final_result = CompareConst.WARNING

    def _compare_ratio(self):
        self.small_value_err_ratio = self._calc_ratio(
            self.npu_precision.get(ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE),
            self.gpu_precision.get(ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE))
        self.rmse_ratio = self._calc_ratio(self.npu_precision.get(ApiPrecisionCompareColumn.RMSE),
                                                      self.gpu_precision.get(ApiPrecisionCompareColumn.RMSE), 10000.0)
        self.max_rel_err_ratio = self._calc_ratio(self.npu_precision.get(ApiPrecisionCompareColumn.MAX_REL_ERR),
                                                self.gpu_precision.get(ApiPrecisionCompareColumn.MAX_REL_ERR), 10000.0)
        self.mean_rel_err_ratio = self._calc_ratio(self.npu_precision.get(ApiPrecisionCompareColumn.MEAN_REL_ERR),
                                                      self.gpu_precision.get(ApiPrecisionCompareColumn.MEAN_REL_ERR))
        self.eb_ratio = self._calc_ratio(self.npu_precision.get(ApiPrecisionCompareColumn.EB),
                                                      self.gpu_precision.get(ApiPrecisionCompareColumn.EB))

    def to_column_value(self):
        return [self.api_name, self.small_value_err_ratio, self.small_value_err_status, self.rmse_ratio, 
        self.rmse_status, self.max_rel_err_ratio, self.max_rel_err_status, self.mean_rel_err_ratio, 
        self.mean_rel_err_status, self.eb_ratio, self.eb_status]

    @staticmethod
    def _get_status(ratio, algorithm):
        error_threshold = benchmark_algorithms_thresholds.get(algorithm).get('error_threshold')
        warning_threshold = benchmark_algorithms_thresholds.get(algorithm).get('warning_threshold')
        if ratio > error_threshold:
            return CompareConst.ERROR
        elif ratio > warning_threshold:
            return CompareConst.WARNING
        return CompareConst.PASS

    @staticmethod
    def _calc_ratio(x, y, default_value=1.0):
        if math.isclose(y, 0.0):
            return 1.0 if math.isclose(x, 0.0) else default_value
        else:
            return abs(x / y)


def write_detail_csv(content, save_path):
    rows = []
    content = ["{:.{}f}".format(item, msCheckerConfig.precision) \
        if isinstance(item, float) else item for item in content]
    rows.append(content)
    write_csv(rows, save_path)


def benchmark_compare(config):
    print_info_log("Start compare task")
    print_info_log(f"Compare task result will be saved in {config.result_csv_path}")
    print_info_log(f"Compare task detail will be saved in {config.details_csv_path}")
    try:
        npu_data = pd.read_csv(config.npu_csv_path)
    except Exception as err:
        print_error_log(f"Open npu csv Error: %s" % str(err))
    check_csv_columns(npu_data.columns, "npu_csv")
    try:
        gpu_data = pd.read_csv(config.gpu_csv_path)
    except Exception as err:
        print_error_log(f"Open gpu csv Error: %s" % str(err))
    check_csv_columns(gpu_data.columns, "gpu_csv")
    detail_csv_title = [ApiPrecisionCompareColumn.get_detail_csv_title()]
    result_csv_title = [ApiPrecisionCompareColumn.get_result_csv_title()]
    write_csv(result_csv_title, config.result_csv_path)
    write_csv(detail_csv_title, config.details_csv_path)
    try:
        analyse_csv(npu_data, gpu_data, config)
    except Exception as err:
        import traceback;traceback.print_exc()
        print_error_log(f"Analyse csv Error: %s" % str(err))
    change_mode(config.result_csv_path, FileCheckConst.DATA_FILE_AUTHORITY)
    change_mode(config.details_csv_path, FileCheckConst.DATA_FILE_AUTHORITY)


def analyse_csv(npu_data, gpu_data, config):
    forward_status, backward_status = [], []
    last_api_name, last_api_dtype = None, None
    for _, row_npu in npu_data.iterrows():
        message = ''
        part_api_name = row_npu[ApiPrecisionCompareColumn.API_NAME]
        row_gpu = gpu_data[gpu_data[ApiPrecisionCompareColumn.API_NAME] == part_api_name]
        api_name, direction_status, _, _ = part_api_name.split(".")
        binary_consistency_check, absolute_threshold_check, benchmark_compare_check = False, False, False
        if row_gpu.empty:
            print_warn_log(f'This API : {part_api_name} does not exist in the GPU data.')
            continue
        if len(row_gpu) > 1:
            msg = f'This API : {part_api_name} has multiple records in the GPU data.'
            raise CompareException(CompareException.INVALID_DATA_ERROR, msg)
        row_gpu = row_gpu.iloc[0]
        if pd.isna(row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE]):
            continue
        _, dedicated_api_name, _ = api_name.split("*")
        new_status = CompareConst.NA
        detail_csv_content = [part_api_name] + [" " for _ in range(17)]
        if row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] not in Binary_Compare_Unsupport_List:
            new_status = check_error_rate(row_npu[ApiPrecisionCompareColumn.ERROR_RATE], 
                                          row_gpu[ApiPrecisionCompareColumn.ERROR_RATE])
            detail_csv_content[14] = new_status
            detail_csv_content[15] = new_status
            detail_csv_content[16] = "二进制一致法"                              
        elif dedicated_api_name in AbsoluteStandardApiName:
            inf_nan_result, rel_err_result, abs_err_result, new_status = get_absolute_threshold_result(row_npu, row_gpu)
            detail_csv_content[11:14] = [inf_nan_result, rel_err_result, abs_err_result]
            detail_csv_content[15] = new_status
            detail_csv_content[16] = "绝对阈值法"
        elif row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] in Benchmark_Compare_Support_List:
            bs = BenchmarkStandard(part_api_name, row_npu, row_gpu)
            bs.get_result()
            new_status = bs.final_result
            detail_csv_content[0:11] = bs.to_column_value()
            detail_csv_content[15] = new_status
            detail_csv_content[16] = "标杆比对法" 

        write_detail_csv(detail_csv_content, config.details_csv_path)

        if last_api_name is not None and api_name != last_api_name:
            if last_api_dtype in Api_Precision_Compare_Unsupport_List:
                message = unsupported_message
                write_csv([[last_api_name, "skip", "skip", message]], config.result_csv_path)
                forward_status, backward_status = [], []
                message = ''
            else:
                forward_result = get_api_checker_result(forward_status)
                backward_result = get_api_checker_result(backward_status)
                write_csv([[last_api_name, forward_result, backward_result, message]], config.result_csv_path)
                forward_status, backward_status = [], []
                message = ''
                
        is_supported = row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] not in Api_Precision_Compare_Unsupport_List
        last_api_name = api_name
        
        last_api_dtype = row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE]
        if not is_supported:
            continue

        if direction_status == 'forward':
            forward_status.append(new_status)
        elif direction_status == 'backward':
            backward_status.append(new_status)
        else:
            print_error_log(f"Invalid direction status: {direction_status}")

    if last_api_name is not None:
        if last_api_dtype in Api_Precision_Compare_Unsupport_List:
            message = unsupported_message
            write_csv([[last_api_name, "skip", "skip", message]], config.result_csv_path)
        else:
            forward_result = get_api_checker_result(forward_status)
            backward_result = get_api_checker_result(backward_status)
            write_csv([[last_api_name, forward_result, backward_result, message]], config.result_csv_path)


def check_error_rate(npu_error_rate, gpu_error_rate):
    return CompareConst.PASS if npu_error_rate == 0 and gpu_error_rate == 0 else CompareConst.ERROR


def get_absolute_threshold_result(row_npu, row_gpu):
    inf_nan_result = CompareConst.PASS if row_npu[ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO] == 0 and \
    row_npu[ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO] == 0 else CompareConst.ERROR
    rel_err_result = CompareConst.PASS if row_npu[ApiPrecisionCompareColumn.REL_ERR_RATIO] == 0 and \
    row_npu[ApiPrecisionCompareColumn.REL_ERR_RATIO] == 0 else CompareConst.ERROR
    abs_err_result = CompareConst.PASS if row_npu[ApiPrecisionCompareColumn.ABS_ERR_RATIO] == 0 and \
    row_npu[ApiPrecisionCompareColumn.ABS_ERR_RATIO] == 0 else CompareConst.ERROR
    if CompareConst.ERROR not in [inf_nan_result, rel_err_result, abs_err_result]:
        absolute_threshold_result = CompareConst.PASS
    else:
        absolute_threshold_result = CompareConst.ERROR

    return inf_nan_result, rel_err_result, abs_err_result, absolute_threshold_result


def get_api_checker_result(status):
    if not status:
        return CompareConst.NA
    for const in (CompareConst.ERROR, CompareConst.WARNING):
        if const in status:
            return const
    return CompareConst.PASS


def check_csv_columns(columns, csv_type):
    required_columns = ApiPrecisionCompareColumn.to_required_columns()
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        msg = f"The followint columns {','.join(missing_columns)} are missing in{csv_type}"
        raise CompareException(CompareException.INVALID_DATA_ERROR, msg)


def _api_precision_compare(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    _api_precision_compare_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    _api_precision_compare_command(args)


def _api_precision_compare_command(args):
    npu_csv_path = get_validated_result_csv_path(args.npu_csv_path, 'detail')
    gpu_csv_path = get_validated_result_csv_path(args.gpu_csv_path, 'detail')
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    check_path_before_create(out_path)
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    result_csv_path = os.path.join(out_path, BENCHMARK_COMPARE_RESULT_FILE_NAME)
    details_csv_path = os.path.join(out_path, BENCHMARK_COMPARE_DETAILS_FILE_NAME)
    compare_config = CompareConfig(npu_csv_path, gpu_csv_path, result_csv_path, details_csv_path)
    benchmark_compare(compare_config)


def _api_precision_compare_parser(parser):
    parser.add_argument("-npu", "--npu_csv_path", dest="npu_csv_path", default="", type=str,
                        help="<Required> , Accuracy_checking_details.csv generated on the NPU by using the "
                             "api_accuracy_checker tool.",
                        required=True)
    parser.add_argument("-gpu", "--gpu_csv_path", dest="gpu_csv_path", default="", type=str,
                        help="<Required> Accuracy_checking_details.csv generated on the GPU by using the "
                             "api_accuracy_checker tool.",
                        required=False)
    parser.add_argument("-o", "--out_path", dest="out_path", default="", type=str,
                        help="<optional> The api precision compare task result out path.",
                        required=False)


if __name__ == '__main__':
    _api_precision_compare()
    print_info_log("Compare task completed.")
    