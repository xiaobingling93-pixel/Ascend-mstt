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

import os
from msprobe.core.common.utils import CompareException
from msprobe.core.common.file_utils import create_directory
from msprobe.core.common.exceptions import FileCheckException
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.compare.ms_compare import ms_compare
from msprobe.core.compare.utils import check_and_return_dir_contents, extract_json
from msprobe.mindspore.compare.ms_graph_compare import GraphMSComparator


def ms_compare_distributed(npu_dump_dir, bench_dump_dir, output_path, **kwargs):
    if kwargs.get('suffix'):
        logger.error("Argument 'suffix' is not supported for compare_distributed.")
        raise CompareException(CompareException.INVALID_PARAM_ERROR)
    is_print_compare_log = kwargs.get('is_print_compare_log', True)
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

        dump_result_param = {
            'npu_json_path': npu_path,
            'bench_json_path': bench_path,
            'is_print_compare_log': is_print_compare_log
        }
        ms_compare(input_param=dump_result_param, output_path=output_path, suffix=f'_{nr}-{br}', **kwargs)


def ms_graph_compare(inputs, outputs):
    try:
        create_directory(outputs)
    except (CompareException, FileCheckException) as error:
        logger.error('Compare failed. Please check the arguments and do it again!')
        return
    ms_comparator = GraphMSComparator(inputs, outputs)
    ms_comparator.compare_core()
