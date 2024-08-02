#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2019-2024. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""
import os
import sys
import re
from msprobe.core.common.utils import CompareException, check_compare_param, \
    check_configuration_param, task_dumppath_get, check_file_or_directory_path, check_regex_prefix_format_valid
from msprobe.pytorch.compare.acc_compare import compare_core
from msprobe.core.common.file_check import create_directory
from msprobe.core.common.exceptions import FileCheckException
from msprobe.pytorch.common.log import logger


def compare_distributed(npu_dump_dir, bench_dump_dir, output_path, **kwargs):
    def check_and_return_dir_contents(dump_dir, prefix):
        """
        check the given dump dir and validate files in dump dir by using the given prefix patterns to build a
        pattern: ^{prefix}(?:0|[0-9][1-9]*)?$

        Args:
            dump_dir (str): dump dir
            prefix (str): prefix for the patterns, prefix should be less than 20 characters and alphanumeric/-/_ only

        Returns:
            content [list]: dir contents
        Raises:
            CompareException: invalid path
            ValueError: prefix not match the patterns

        """
        check_regex_prefix_format_valid(prefix)
        check_file_or_directory_path(dump_dir, True)
        contents = os.listdir(dump_dir)
        pattern = re.compile(rf'^{prefix}(?:0|[0-9][1-9]*)?$')
        for name in contents:
            if not pattern.match(name):
                logger.error(
                    f"dump_dir contains '{name}'. Expected '{prefix}'. This name is not in the format of dump "
                    f"output. Please check and delete irrelevant files in {dump_dir} and try again."
                )
                raise CompareException(CompareException.INVALID_PATH_ERROR)
        return contents

    def extract_json(dirname, stack_json=False):
        json_path = ''
        for fname in os.listdir(dirname):
            full_path = os.path.join(dirname, fname)
            if full_path.endswith('.json'):
                json_path = full_path
                if not stack_json and 'stack' not in json_path:
                    break
                if stack_json and 'stack' in json_path:
                    break

        # Provide robustness on invalid directory inputs
        if not json_path:
            logger.error(f'No file is found in dump dir {dirname}. ')
            raise CompareException(CompareException.NO_DUMP_FILE_ERROR)
        return json_path

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
        npu_json_path = extract_json(npu_data_dir, stack_json=False)
        bench_json_path = extract_json(bench_data_dir, stack_json=False)
        stack_json_path = extract_json(npu_data_dir, stack_json=True)

        dump_result_param = {
            'npu_path': npu_path,
            'bench_path': bench_path,
            'stack_path': stack_path,
            'is_print_compare_log': True
        }
        try:
            summary_compare, md5_compare = task_dumppath_get(dump_result_param)
            check_configuration_param(stack_mode, auto_analyze, fuzzy_match)
            create_directory(output_path)
            check_compare_param(dump_result_param, output_path, summary_compare=summary_compare, md5_compare=md5_compare)
        except (CompareException, FileCheckException) as error:
            logger.error('Compare failed. Please check the arguments and do it again!')
            sys.exit(error.code)
        compare_core(dump_result_param, output_path, suffix=f'_{nr}-{br}', summary_compare=summary_compare,
                     md5_compare=md5_compare, **kwargs)
