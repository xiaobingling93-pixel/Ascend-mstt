import os
from msprobe.core.common.utils import CompareException, check_compare_param, \
    check_configuration_param, task_dumppath_get
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.exceptions import FileCheckException
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.compare.ms_compare import MSComparator
from msprobe.core.compare.utils import check_and_return_dir_contents, extract_json
from msprobe.mindspore.compare.ms_graph_compare import GraphMSComparator


def ms_compare_distributed(npu_dump_dir, bench_dump_dir, output_path, **kwargs):
    if kwargs.get('suffix'):
        logger.error("Argument 'suffix' is not supported for compare_distributed.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    stack_mode = kwargs.get('stack_mode', False)
    auto_analyze = kwargs.get('auto_analyze', True)
    fuzzy_match = kwargs.get('fuzzy_match', False)
    # get the ranks and match by order
    npu_ranks = sorted(check_and_return_dir_contents(npu_dump_dir, 'rank'))
    bench_ranks = sorted(check_and_return_dir_contents(bench_dump_dir, 'rank'))
    if len(npu_ranks) != len(bench_ranks):
        logger.error('The number of ranks in the two runs are different. '
                        'Unable to match the ranks. Please use another folder to compare '
                        'or use compare() api and manually match the ranks.')
        raise CompareException(CompareException.INVALID_PATH_ERROR)
    for nr, br in zip(npu_ranks, bench_ranks):
        npu_data_dir = os.path.join(npu_dump_dir, nr)
        bench_data_dir = os.path.join(bench_dump_dir, br)
        npu_path = extract_json(npu_data_dir, stack_json=False)
        bench_path = extract_json(bench_data_dir, stack_json=False)
        stack_path = extract_json(npu_data_dir, stack_json=True)

        dump_result_param = {
            'npu_json_path': npu_path,
            'bench_json_path': bench_path,
            'stack_json_path': stack_path,
            'is_print_compare_log': True
        }
        try:
            summary_compare, md5_compare = task_dumppath_get(dump_result_param)
            check_configuration_param(stack_mode, auto_analyze, fuzzy_match, 
                                      dump_result_param.get('is_print_compare_log', True))
            create_directory(output_path)
            check_compare_param(dump_result_param, output_path, 
                                summary_compare=summary_compare, md5_compare=md5_compare)
        except (CompareException, FileCheckException) as error:
            logger.error('Compare failed. Please check the arguments and do it again!')
            raise CompareException(error.code) from error
        ms_comparator = MSComparator()
        ms_comparator.compare_core(dump_result_param, output_path, suffix=f'_{nr}-{br}', 
                                   summary_compare=summary_compare, md5_compare=md5_compare, **kwargs)


def ms_graph_compare(inputs, outputs):
    try:
        create_directory(outputs)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        return
    ms_comparator = GraphMSComparator(inputs, outputs)
    ms_comparator.compare_core()
