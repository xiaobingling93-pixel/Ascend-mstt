# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException
from msprobe.core.common.const import CompareConst


def _handle_multi_process(func, input_parma, result_df, lock):
    process_num = int((multiprocessing.cpu_count() + 1) / 2)
    op_name_mapping_dict = read_dump_data(result_df)

    df_chunk_size = len(result_df) // process_num
    if df_chunk_size > 0:
        df_chunks = [result_df.iloc[i:i + df_chunk_size] for i in range(0, len(result_df), df_chunk_size)]
    else:
        df_chunks = [result_df]

    results = []
    pool = multiprocessing.Pool(process_num)

    def err_call(args):
        logger.error('multiprocess compare failed! Reason: {}'.format(args))
        try:
            pool.terminate()
        except OSError as e:
            logger.error("pool terminate failed")

    progress_bar = tqdm(total=len(result_df), desc="API/Module Item Compare Process", unit="row", ncols=100)

    def update_progress(size, progress_lock):
        with progress_lock:
            progress_bar.update(size)

    for process_idx, df_chunk in enumerate(df_chunks):
        idx = df_chunk_size * process_idx
        chunk_size = len(df_chunk)
        result = pool.apply_async(func,
                                  args=(idx, op_name_mapping_dict, df_chunk, lock, input_parma),
                                  error_callback=err_call,
                                  callback=update_progress(chunk_size, lock))
        results.append(result)
    final_results = [r.get() for r in results]
    pool.close()
    pool.join()
    return pd.concat(final_results, ignore_index=True)


def _ms_graph_handle_multi_process(func, result_df, mode):
    process_num = int((multiprocessing.cpu_count() + 1) // 4)
    df_chunk_size = len(result_df) // process_num
    if df_chunk_size > 0:
        df_chunks = [result_df.iloc[i:i + df_chunk_size] for i in range(0, len(result_df), df_chunk_size)]
    else:
        df_chunks = [result_df]

    results = []
    pool = multiprocessing.Pool(process_num)

    def err_call(args):
        logger.error('multiprocess compare failed! Reason: {}'.format(args))
        try:
            pool.terminate()
        except OSError as e:
            logger.error("pool terminate failed")

    for df_chunk in df_chunks:
        result = pool.apply_async(func, args=(df_chunk, mode), error_callback=err_call)
        results.append(result)
    final_results = [r.get() for r in results]
    pool.close()
    pool.join()
    return pd.concat(final_results, ignore_index=True)


def read_dump_data(result_df):
    try:
        npu_dump_name_list = result_df.iloc[0:, 0].tolist()
        npu_dump_tensor_list = result_df.iloc[0:, -1].tolist()
        op_name_mapping_dict = {}
        for index, _ in enumerate(npu_dump_name_list):
            npu_dump_name = npu_dump_name_list[index]
            npu_dump_tensor = npu_dump_tensor_list[index]
            op_name_mapping_dict[npu_dump_name] = [npu_dump_tensor, npu_dump_tensor]
        return op_name_mapping_dict
    except ValueError as e:
        logger.error('result dataframe is not found.')
        raise CompareException(CompareException.INVALID_DATA_ERROR) from e
    except IndexError as e:
        logger.error('result dataframe elements can not be access.')
        raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from e


@dataclass
class ComparisonResult:
    cos_result: list
    max_err_result:  list
    max_relative_err_result: list
    err_msgs: list
    one_thousand_err_ratio_result: list
    five_thousand_err_ratio_result: list


def _save_cmp_result(offset, result: ComparisonResult, result_df, lock):
    """
        Save comparison results into the result DataFrame with thread safety.
    Args:
        offset: offset for index
        result: data struct of ComparisonResult
        result_df: result of DataFrame
        lock: thread lock

    Returns:
        comparison results in DataFrame
    """

    lock.acquire()
    try:
        for i, _ in enumerate(result.cos_result):
            process_index = i + offset
            result_df.loc[process_index, CompareConst.COSINE] = result.cos_result[i]
            result_df.loc[process_index, CompareConst.MAX_ABS_ERR] = result.max_err_result[i]
            result_df.loc[process_index, CompareConst.MAX_RELATIVE_ERR] = result.max_relative_err_result[i]
            result_df.loc[process_index, CompareConst.ERROR_MESSAGE] = result.err_msgs[i]
            result_df.loc[process_index, CompareConst.ACCURACY] = (
                check_accuracy(result.cos_result[i], result.max_err_result[i]))
            result_df.loc[process_index, CompareConst.ONE_THOUSANDTH_ERR_RATIO] = (
                result.one_thousand_err_ratio_result)[i]
            result_df.loc[process_index, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO] = (
                result.five_thousand_err_ratio_result)[i]
        return result_df
    except ValueError as e:
        logger.error('result dataframe is not found.')
        raise CompareException(CompareException.INVALID_DATA_ERROR) from e
    except IndexError as e:
        logger.error('result dataframe elements can not be access.')
        raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from e
    finally:
        lock.release()
        
        
def check_accuracy(cos, max_abs_err):
    if cos == CompareConst.SHAPE_UNMATCH:
        return CompareConst.ACCURACY_CHECK_UNMATCH
    if cos == CompareConst.NONE or max_abs_err == CompareConst.NONE:
        return CompareConst.NONE
    if cos == "N/A" or max_abs_err == "N/A":
        return CompareConst.ACCURACY_CHECK_NO
    try:
        cos, max_abs_err = float(cos), float(max_abs_err)
    except ValueError:
        logger.warning("Cosine or MaxAbsErr can not get float value.")
        return CompareConst.NONE
    if cos < CompareConst.COS_THRESHOLD and max_abs_err > CompareConst.MAX_ABS_ERR_THRESHOLD:
        return CompareConst.ACCURACY_CHECK_NO
    if cos < CompareConst.COS_MAX_THRESHOLD or max_abs_err > CompareConst.MAX_ABS_ERR_MAX_THRESHOLD:
        return CompareConst.ACCURACY_CHECK_NO
    return CompareConst.ACCURACY_CHECK_YES
