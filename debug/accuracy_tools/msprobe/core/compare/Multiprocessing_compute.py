
import multiprocessing
import pandas as pd
from msprobe.core.common.log import logger
from msprobe.core.common.utils import  CompareException



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

    for process_idx, df_chunk in enumerate(df_chunks):
        idx = df_chunk_size * process_idx
        result = pool.apply_async(func,
                                  args=(idx, op_name_mapping_dict, df_chunk, lock, input_parma),
                                  error_callback=err_call)
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