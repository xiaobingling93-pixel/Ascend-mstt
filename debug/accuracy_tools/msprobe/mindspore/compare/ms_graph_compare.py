import csv
import glob
import os
import sys
import copy

import numpy as np
import pandas as pd
from msprobe.core.common.const import CompareConst, GraphMode
from msprobe.core.common.exceptions import FileCheckException
from msprobe.core.common.file_check import create_directory
from msprobe.mindspore.common.log import logger
from msprobe.core.common.utils import add_time_with_xlsx, CompareException, load_npy
from msprobe.core.compare.multiprocessing_compute import _ms_graph_handle_multi_process, check_accuracy
from msprobe.core.compare.npy_compare import npy_data_check, statistics_data_check, reshape_value, compare_ops_apply
from msprobe.core.common.file_check import FileOpen

class row_data:
    def __init__(self, mode):
        self.basic_data = copy.deepcopy(CompareConst.MS_GRAPH_BASE)
        self.npy_data = copy.deepcopy(CompareConst.MS_GRAPH_NPY)
        self.statistic_data = copy.deepcopy(CompareConst.MS_GRAPH_STATISTIC)
        if mode == GraphMode.NPY_MODE:
            self.data = {**self.basic_data, **self.npy_data}
        else:
            self.data = {**self.basic_data, **self.statistic_data}

    def __call__(self):
        return self.data


def generate_step(npu_path, rank_id):
    step_set = set()
    rank_path = os.path.join(npu_path, f"rank_{rank_id}")
    if not os.path.exists(rank_path):
        return []
    for path in os.listdir(rank_path):
        if path not in ["execution_order", "graphs"]:
            data_path = os.path.join(rank_path, path)
            for graph_path in os.listdir(data_path):
                step_set.update([int(i) for i in os.listdir(os.path.join(data_path, graph_path))])
    return sorted(step_set)


def generate_path_by_rank_step(base_path, rank_id, step_id):
    path_with_rank_id = os.path.join(base_path, f"rank_{rank_id}")
    if not os.path.exists(path_with_rank_id):
        return ''
    for path in os.listdir(path_with_rank_id):
        if path not in ["execution_order", "graphs"]:

            return os.path.join(path_with_rank_id, path, "*", str(step_id))
    logger.error(f"Data_path {path_with_rank_id} is not exist.")
    return ''


def statistic_data_read(statistic_file_list, statistic_file_path):
    data_list = []
    statistic_data_list = []
    for statistic_file in statistic_file_list:
        with FileOpen(statistic_file, "r") as f:
            csv_reader = csv.reader(f, delimiter=",")
            header = next(csv_reader)
            header_index = {'Data Type': None, 'Shape': None, 'Max Value': None, 'Min Value': None,
                            'Avg Value': None, 'L2Norm Value': None}
            for key in header_index.keys():
                for index, value in enumerate(header):
                    if key == value:
                        header_index[key] = index
            for key in header_index.keys():
                if header_index[key] is None:
                    logger.error(f"Data_path {statistic_file_path} has no key {key}")
                    raise FileCheckException(f"Data_path {statistic_file_path} has no key {key}")
            statistic_data_list.extend([row for row in csv_reader])

    for data in statistic_data_list:
        compare_key = f"{data[1]}.{data[2]}.{data[3]}.{data[5]}"
        timestamp = int(data[4])
        data_list.append(
            [statistic_file_path, compare_key, timestamp, data[header_index['Data Type']],
             data[header_index['Shape']], data[header_index['Max Value']], data[header_index['Min Value']],
             data[header_index['Avg Value']], data[header_index['L2Norm Value']]])
    return data_list


def generate_data_name(data_path):
    data_list = []

    mapping_path = os.path.join(data_path, "mapping.csv")
    statistic_path = os.path.join(data_path, "statistic.csv")
    npy_path = os.path.join(data_path, "*.npy")

    mapping_file_list = glob.glob(mapping_path)
    statistic_file_list = glob.glob(statistic_path)
    npy_file_list = glob.glob(npy_path)

    mapping_exist = bool(mapping_file_list)
    statistic_exist = bool(statistic_file_list)
    npy_exist = bool(npy_file_list)

    mapping_dict = []
    if mapping_exist:
        for mapping_file in mapping_file_list:
            with FileOpen(mapping_file, "r") as f:
                csv_reader = csv.reader(f, delimiter=",")
                header = next(csv_reader)
                for row in csv_reader:
                    mapping_dict[row[0]] = row[1]

    if npy_exist:
        for data in npy_file_list:
            if data in mapping_dict:
                split_list = mapping_dict[data].split(".")
            else:
                split_list = data.split(".")
            compare_key = f"{split_list[1]}.{split_list[2]}.{split_list[3]}.{split_list[5]}.{split_list[6]}"
            timestamp = int(split_list[4])

            data_list.append([os.path.join(data_path, data), compare_key, timestamp])
    elif statistic_exist:
        data_list = statistic_data_read(statistic_file_list, os.path.join(data_path, statistic_path))

    if npy_exist:
        mode = GraphMode.NPY_MODE
    elif statistic_exist:
        mode = GraphMode.STATISTIC_MODE
    else:
        mode = GraphMode.ERROR_MODE
        logger.error(f"Error mode.")
    return mode, data_list


class GraphMSComparator:
    def __init__(self, input_param, output_path):
        self.output_path = output_path
        self.base_npu_path = input_param.get('npu_path', None)
        self.base_bench_path = input_param.get('bench_path', None)
        self.rank_list = input_param.get('rank_id', [])
        self.step_list = input_param.get('step_id', [])

    @staticmethod
    def compare_ops(compare_result_db, mode):

        def npy_mode_compute(row):
            result_dict = row_data(GraphMode.NPY_MODE)()

            def process_npy_file(file_path, name_prefix, result):
                if os.path.exists(file_path):
                    data = load_npy(file_path)
                    result[f'{name_prefix} Name'] = file_path
                    result[f'{name_prefix} Dtype'] = data.dtype
                    result[f'{name_prefix} Tensor Shape'] = data.shape
                    result[f'{name_prefix} max'] = np.max(data)
                    result[f'{name_prefix} min'] = np.min(data)
                    result[f'{name_prefix} mean'] = np.mean(data)
                    result[f'{name_prefix} l2norm'] = np.linalg.norm(data)
                    return data
                return ""

            n_value = process_npy_file(row[CompareConst.NPU_NAME], 'NPU', result_dict)
            b_value = process_npy_file(row[CompareConst.BENCH_NAME], 'Bench', result_dict)

            error_flag, error_message = npy_data_check(n_value, b_value)
            result_dict[CompareConst.ERROR_MESSAGE] = error_message

            if not error_flag:
                n_value, b_value = reshape_value(n_value, b_value)
                result_list, err_msg = compare_ops_apply(n_value, b_value, False, "")
                result_dict[CompareConst.COSINE] = result_list[0]
                result_dict[CompareConst.MAX_ABS_ERR] = result_list[1]
                result_dict[CompareConst.MAX_RELATIVE_ERR] = result_list[2]
                result_dict[CompareConst.ONE_THOUSANDTH_ERR_RATIO] = result_list[3]
                result_dict[CompareConst.FIVE_THOUSANDTHS_ERR_RATIO] = result_list[4]
                result_dict[CompareConst.ACCURACY] = check_accuracy(result_list[0], result_list[1])
                result_dict[CompareConst.ERROR_MESSAGE] = err_msg

            return pd.Series(result_dict)

        def statistic_mode_compute(row):
            result_dict = row_data('STATISTIC')()

            def update_result_dict(result, rows, prefix):
                result[f'{prefix} Name'] = rows[f'{prefix} Name']
                result[f'{prefix} Dtype'] = rows[f'{prefix} Dtype']
                result[f'{prefix} Tensor Shape'] = rows[f'{prefix} Tensor Shape']
                result[f'{prefix} max'] = np.float32(rows[f'{prefix} max'])
                result[f'{prefix} min'] = np.float32(rows[f'{prefix} min'])
                result[f'{prefix} mean'] = np.float32(rows[f'{prefix} mean'])
                result[f'{prefix} l2norm'] = np.float32(rows[f'{prefix} l2norm'])

            # 使用示例
            update_result_dict(result_dict, row, 'NPU')
            update_result_dict(result_dict, row, 'Bench')
            error_flag, error_message = statistics_data_check(result_dict)
            result_dict[CompareConst.ERROR_MESSAGE] += error_message
            if not error_flag:
                result_dict[CompareConst.MAX_DIFF] = np.abs(
                    result_dict[CompareConst.NPU_MAX] - result_dict[CompareConst.BENCH_MAX])
                result_dict[CompareConst.MIN_DIFF] = np.abs(
                    result_dict[CompareConst.NPU_MIN] - result_dict[CompareConst.BENCH_MIN])
                result_dict[CompareConst.MEAN_DIFF] = np.abs(
                    result_dict[CompareConst.NPU_MEAN] - result_dict[CompareConst.BENCH_MEAN])
                result_dict[CompareConst.NORM_DIFF] = np.abs(
                    result_dict[CompareConst.NPU_NORM] - result_dict[CompareConst.BENCH_NORM])
                result_dict[CompareConst.MAX_RELATIVE_ERR] = result_dict[CompareConst.MAX_DIFF] / result_dict[
                    CompareConst.BENCH_MAX] if result_dict[CompareConst.BENCH_MAX] > 0 else 0
                result_dict[CompareConst.MAX_RELATIVE_ERR] = str(result_dict[CompareConst.MAX_RELATIVE_ERR] * 100) + "%"
                result_dict[CompareConst.MIN_RELATIVE_ERR] = result_dict[CompareConst.MIN_DIFF] / result_dict[
                    CompareConst.BENCH_MIN] if result_dict[CompareConst.BENCH_MIN] > 0 else 0
                result_dict[CompareConst.MIN_RELATIVE_ERR] = str(result_dict[CompareConst.MIN_RELATIVE_ERR] * 100) + "%"
                result_dict[CompareConst.MEAN_RELATIVE_ERR] = result_dict[CompareConst.MEAN_DIFF] / result_dict[
                    CompareConst.BENCH_MEAN] if result_dict[CompareConst.BENCH_MEAN] > 0 else 0
                result_dict[CompareConst.MEAN_RELATIVE_ERR] = str(
                    result_dict[CompareConst.MEAN_RELATIVE_ERR] * 100) + "%"
                result_dict[CompareConst.NORM_RELATIVE_ERR] = result_dict[CompareConst.NORM_DIFF] / result_dict[
                    CompareConst.BENCH_NORM] if result_dict[CompareConst.BENCH_NORM] > 0 else 0
                result_dict[CompareConst.NORM_RELATIVE_ERR] = str(
                    result_dict[CompareConst.NORM_RELATIVE_ERR] * 100) + "%"
                magnitude_diff = result_dict[CompareConst.MAX_DIFF] / (
                            max(result_dict[CompareConst.NPU_MAX], result_dict[CompareConst.BENCH_MAX]) + 1e-10)
                if magnitude_diff > CompareConst.MAGNITUDE:
                    result_dict[CompareConst.ACCURACY] = 'No'
                else:
                    result_dict[CompareConst.ACCURACY] = 'Yes'

            return pd.Series(result_dict)

        if mode == GraphMode.NPY_MODE:
            compare_result_db = compare_result_db.apply(npy_mode_compute, axis=1)
        else:
            compare_result_db = compare_result_db.apply(statistic_mode_compute, axis=1)
        return compare_result_db

    def compare_core(self):
        logger.info("Please check whether the input data belongs to you. If not, there may be security risks.")

        # split by rank and step
        if not self.rank_list:
            self.rank_list = [int(i.split("_")[-1]) for i in os.listdir(self.base_npu_path)]
        for rank_id in self.rank_list:
            if not self.step_list:
                self.step_list = generate_step(self.base_npu_path, rank_id)
            for step_id in self.step_list:
                compare_result_df, mode = self.compare_process(rank_id, step_id)
                if isinstance(compare_result_df, list):
                    is_empty = not compare_result_df
                elif isinstance(compare_result_df, pd.DataFrame):
                    is_empty = compare_result_df.empty
                else:
                    is_empty = True
                if is_empty or not mode:
                    continue
                compare_result_df = self._do_multi_process(compare_result_df, mode)
                compare_result_name = add_time_with_xlsx(f"compare_result_{str(rank_id)}_{str(step_id)}")
                compare_result_path = os.path.join(os.path.realpath(self.output_path), f"{compare_result_name}")
                compare_result_df.to_excel(compare_result_path, index=False)
                logger.info(f"Compare rank: {rank_id} step: {step_id} finish. Compare result: {compare_result_path}.")

    def compare_process(self, rank_id, step_id):
        # generate data_path
        npu_data_path = generate_path_by_rank_step(self.base_npu_path, rank_id, step_id)
        bench_data_path = generate_path_by_rank_step(self.base_bench_path, rank_id, step_id)
        if not npu_data_path or not bench_data_path:
            return [], ''

        # generate file name
        npu_mode, npu_data_list = generate_data_name(npu_data_path)
        match_mode, match_data_list = generate_data_name(bench_data_path)

        if npu_mode == "ERROR_MODE" or match_mode == "ERROR_MODE":
            logger.warning(f"Data_path {npu_data_path} or {bench_data_path} is not exist.")
            return [], ''
        if npu_mode != match_mode:
            logger.error(f"NPU mode {npu_mode} not equal to MATCH mode {match_mode}.")
            return [], ''

        if npu_mode == 'NPY_MODE':
            npu_data_df = pd.DataFrame(npu_data_list, columns=[CompareConst.NPU_NAME, 'Compare Key', 'TimeStamp'])
            bench_data_df = pd.DataFrame(match_data_list, columns=[CompareConst.BENCH_NAME, 'Compare Key', 'TimeStamp'])
        else:
            npu_data_df = pd.DataFrame(npu_data_list,
                                       columns=[CompareConst.NPU_NAME, 'Compare Key', 'TimeStamp', CompareConst.NPU_DTYPE, CompareConst.NPU_SHAPE,
                                                CompareConst.NPU_MAX, CompareConst.NPU_MIN, CompareConst.NPU_MEAN, CompareConst.NPU_NORM])
            bench_data_df = pd.DataFrame(match_data_list,
                                         columns=[CompareConst.BENCH_NAME, 'Compare Key', 'TimeStamp', CompareConst.BENCH_DTYPE,
                                                  CompareConst.BENCH_SHAPE, CompareConst.BENCH_MAX, CompareConst.BENCH_MIN, CompareConst.BENCH_MEAN,
                                                  CompareConst.BENCH_NORM])

        npu_data_df['Local Index'] = npu_data_df.sort_values('TimeStamp').groupby('Compare Key').cumcount()
        bench_data_df['Local Index'] = bench_data_df.sort_values('TimeStamp').groupby('Compare Key').cumcount()

        compare_result_df = pd.merge(npu_data_df, bench_data_df, on=['Compare Key', 'Local Index'], how='outer')

        compare_result_df[CompareConst.NPU_NAME] = compare_result_df[CompareConst.NPU_NAME].fillna('')
        compare_result_df[CompareConst.BENCH_NAME] = compare_result_df[CompareConst.BENCH_NAME].fillna('')
        
        return compare_result_df, npu_mode

    def _do_multi_process(self, result_df, mode):
        try:
            result_df = _ms_graph_handle_multi_process(self.compare_ops, result_df, mode)
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e
        return result_df
