import json
import os

from msprobe.core.common.const import Const, CompareConst, MsCompareConst
from msprobe.core.common.file_utils import FileOpen, create_directory, write_csv
from msprobe.core.common.utils import add_time_as_suffix

from msprobe.mindspore.api_accuracy_checker.api_info import ApiInfo
from msprobe.mindspore.api_accuracy_checker.api_runner import api_runner, ApiInputAggregation
from msprobe.mindspore.api_accuracy_checker.base_compare_algorithm import compare_algorithms
# from msprobe.mindspore.api_accuracy_checker.data_manager import DataManager
from msprobe.mindspore.api_accuracy_checker.utils import (check_and_get_from_json_dict, global_context,
                                                          trim_output_compute_element_list)
from msprobe.mindspore.common.log import logger


class ResultCsvEntry:
    def __init__(self) -> None:
        self.forward_pass_status = None
        self.backward_pass_status = None
        self.forward_err_msg = ""
        self.backward_err_msg = ""
        self.overall_err_msg = None

        # 需要转换为绝对路径


class DataManager:
    def __init__(self, csv_dir, result_csv_path):
        self.results = {}
        self.is_first_write = True  # 标记用于添加表头
        # self.detail_out_path = None
        # self.result_out_path = None
        self.csv_dir = csv_dir
        self.api_names_set = set()  # 存储已经出现的 API 名称的集合
        # 如果传入了 result_csv_path，则启用断点续检
        if result_csv_path:
            self.resume_from_last_csv(result_csv_path)
            self.initialize_api_names_set(result_csv_path)
        else:
            # 默认情况下，设置输出路径为空，等待首次写入时初始化
            self.detail_out_path = None
            self.result_out_path = None

    def initialize_api_names_set(self, result_csv_path):
        """读取现有的 CSV 文件并存储已经出现的 API 名称到集合中"""
        import csv
        with open(result_csv_path, mode='r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            headers = next(reader, None)  # 读取标题行

            # 打印标题行内容以进行调试
            print("Headers found in CSV:", headers)  # 调试输出
            logger.info(f"Headers found in CSV: {headers}")
            # 获取 "API Name" 列的索引
            api_name_index = None
            for i, header in enumerate(headers):
                if "API Name" in header:  # CSV 文件的标题行包含了字节顺序标记,所以直接使用通过包含方式来查找
                    api_name_index = i
                    break

            if api_name_index is None:
                print("[ERROR] No column contains 'API Name'.")
                return

                # 读取每一行的 API 名称
            for row in reader:
                if row and len(row) > api_name_index:
                    api_name = row[api_name_index]
                    if api_name:
                        self.api_names_set.add(api_name)

            logger.warning(f"Initialized API names set from existing CSV: {self.api_names_set}")

    def is_unique_api(self, api_name):
        """检查 API 名称是否唯一，如果已经存在则返回 False，否则加入集合并返回 True"""
        if api_name in self.api_names_set:
            return False
        else:
            self.api_names_set.add(api_name)
            return True

    def resume_from_last_csv(self, result_csv_path):
        """从上次运行的 result_csv_path 恢复断点"""
        # 获取上次的目录路径
        last_dir = os.path.dirname(result_csv_path)

        # 设置当前目录和输出路径，确保在首次写入时使用
        self.csv_dir = last_dir
        self.detail_out_path = os.path.join(last_dir, os.path.basename(result_csv_path).replace("result", "details"))
        self.result_out_path = result_csv_path
        self.is_first_write = False  # 设置为 False，避免重复写入表头

        logger.info(f"Resuming from csv. Using last directory: {self.csv_dir}")
        logger.info(f"Detail output path set to: {self.detail_out_path}")
        logger.info(f"Result output path set to: {self.result_out_path}")

    def save_results(self, api_name_str):
        if self.is_first_write:
            self.detail_out_path = os.path.join(self.csv_dir, add_time_as_suffix(MsCompareConst.DETAIL_CSV_FILE_NAME))
            self.result_out_path = os.path.join(self.csv_dir, add_time_as_suffix(MsCompareConst.RESULT_CSV_FILE_NAME))

            # 直接写入表头
            logger.info("Writing CSV headers for the first time.")
            self.write_csv_header_if_first_time(self.detail_out_path, self.get_detail_csv_header)
            self.write_csv_header_if_first_time(self.result_out_path, self.get_result_csv_header)
            self.is_first_write = False  # 写入后标记为 False，避免重复写入表头

        """写入详细输出和结果摘要并清理结果"""
        logger.warning("Starting to write detailed output to CSV.")
        self.to_detail_csv(self.detail_out_path)
        logger.warning(f"Detailed output for {api_name_str} written to {self.detail_out_path}.")

        logger.warning("Starting to write result summary to CSV.")
        self.to_result_csv(self.result_out_path)
        logger.warning(f"Result summary for {api_name_str} written to {self.result_out_path}.")

        # 清理记录，准备下一次调用
        self.clear_results()

    def record(self, output_list):
        if output_list is None:
            return
        for output in output_list:
            api_real_name, forward_or_backward, basic_info, compare_result_dict = output
            key = (api_real_name, forward_or_backward)
            if key not in self.results:
                self.results[key] = []
            self.results[key].append((basic_info, compare_result_dict))
            logger.debug(f"Updated self.results for key {key}: {self.results[key]}")
        logger.debug(f"Complete self.results after recording: {self.results}")

    def clear_results(self):
        """清空 self.results 数据"""
        logger.debug("Clearing self.results data.")
        self.results.clear()

    def get_detail_csv_header(self):
        """获取详细 CSV 文件的表头"""
        detail_csv_header_basic_info = [
            MsCompareConst.DETAIL_CSV_API_NAME,
            MsCompareConst.DETAIL_CSV_BENCH_DTYPE,
            MsCompareConst.DETAIL_CSV_TESTED_DTYPE,
            MsCompareConst.DETAIL_CSV_SHAPE,
        ]
        detail_csv_header_compare_result = list(compare_algorithms.keys())
        detail_csv_header_status = [
            MsCompareConst.DETAIL_CSV_PASS_STATUS,
            MsCompareConst.DETAIL_CSV_MESSAGE,
        ]
        return detail_csv_header_basic_info + detail_csv_header_compare_result + detail_csv_header_status

    def get_result_csv_header(self):
        """获取结果 CSV 文件的表头"""
        return [
            MsCompareConst.DETAIL_CSV_API_NAME,
            MsCompareConst.RESULT_CSV_FORWARD_TEST_SUCCESS,
            MsCompareConst.RESULT_CSV_BACKWARD_TEST_SUCCESS,
            MsCompareConst.DETAIL_CSV_MESSAGE,
        ]

    def write_csv_header_if_first_time(self, csv_path, header_func):
        """如果是第一次写入，则写入 CSV 表头"""
        header = header_func()  # 获取表头
        logger.debug(f"Writing CSV header: {header}")
        write_csv([header], csv_path, mode="a+")

    def to_detail_csv(self, csv_dir):
        logger.info("Preparing detail CSV headers and rows.")
        detail_csv = []

        detail_csv_header_compare_result = list(compare_algorithms.keys())
        # if self.is_first_write:
        #     detail_csv_header_basic_info = [
        #         MsCompareConst.DETAIL_CSV_API_NAME,
        #         MsCompareConst.DETAIL_CSV_BENCH_DTYPE,
        #         MsCompareConst.DETAIL_CSV_TESTED_DTYPE,
        #         MsCompareConst.DETAIL_CSV_SHAPE,
        #     ]
        #     detail_csv_header_status = [
        #         MsCompareConst.DETAIL_CSV_PASS_STATUS,
        #         MsCompareConst.DETAIL_CSV_MESSAGE,
        #     ]
        #     detail_csv_header = detail_csv_header_basic_info + detail_csv_header_compare_result + detail_csv_header_status
        #     detail_csv.append(detail_csv_header)
        #     logger.debug(f"Detail CSV headers: {detail_csv_header}")

        for _, results in self.results.items():
            for res in results:
                basic_info, compare_result_dict = res
                csv_row_basic_info = [
                    basic_info.api_name,
                    basic_info.bench_dtype,
                    basic_info.tested_dtype,
                    basic_info.shape
                ]
                csv_row_compare_result = [
                    compare_result_dict.get(algorithm_name).compare_value
                    for algorithm_name in detail_csv_header_compare_result
                ]
                csv_row_status = [basic_info.status, basic_info.err_msg]
                csv_row = csv_row_basic_info + csv_row_compare_result + csv_row_status
                detail_csv.append(csv_row)
                logger.debug(f"Detail CSV row added: {csv_row}")

        logger.info(f"Writing detail CSV to {csv_dir}.")
        write_csv(detail_csv, csv_dir, mode="a+")
        logger.info(f"Detail CSV written successfully to {csv_dir}.")

    def to_result_csv(self, csv_dir):
        logger.info("Preparing result CSV data.")
        result_csv = []
        # if self.is_first_write:
        #     result_csv_header = [
        #         MsCompareConst.DETAIL_CSV_API_NAME,
        #         MsCompareConst.RESULT_CSV_FORWARD_TEST_SUCCESS,
        #         MsCompareConst.RESULT_CSV_BACKWARD_TEST_SUCCESS,
        #         MsCompareConst.DETAIL_CSV_MESSAGE,
        #     ]
        #     result_csv.append(result_csv_header)
        #     logger.debug(f"Result CSV headers: {result_csv_header}")

        result_csv_dict = {}
        for key, results in self.results.items():
            api_real_name, forward_or_backward = key
            pass_status = CompareConst.PASS
            overall_err_msg = ""

            for res in results:
                basic_info, _ = res
                if basic_info.status != CompareConst.PASS:
                    pass_status = CompareConst.ERROR
                overall_err_msg += basic_info.err_msg

            overall_err_msg = "" if pass_status == CompareConst.PASS else overall_err_msg

            if api_real_name not in result_csv_dict:
                result_csv_dict[api_real_name] = ResultCsvEntry()
            if forward_or_backward == Const.FORWARD:
                result_csv_dict[api_real_name].forward_pass_status = pass_status
                result_csv_dict[api_real_name].forward_err_msg = overall_err_msg
            else:
                result_csv_dict[api_real_name].backward_pass_status = pass_status
                result_csv_dict[api_real_name].backward_err_msg = overall_err_msg

        for api_name, entry in result_csv_dict.items():
            overall_err_msg = "" if (entry.forward_pass_status == CompareConst.PASS and
                                     entry.backward_pass_status == CompareConst.PASS) else \
                entry.forward_err_msg + entry.backward_err_msg
            row = [
                api_name,
                entry.forward_pass_status,
                entry.backward_pass_status,
                overall_err_msg
            ]
            result_csv.append(row)
            logger.debug(f"Result CSV row added: {row}")

        logger.info(f"Writing result CSV to {csv_dir}.")
        write_csv(result_csv, csv_dir, mode="a+")
        logger.info(f"Result CSV written successfully to {csv_dir}.")

        # 设置标记为 False，防止后续重复添加表头
        self.is_first_write = False
