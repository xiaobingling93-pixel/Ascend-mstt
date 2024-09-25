#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import subprocess
import json
import os
import sys
import argparse
import time
import signal
import threading
from collections import namedtuple
from itertools import cycle
from tqdm import tqdm
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut import _run_ut_parser, preprocess_forward_content
from msprobe.pytorch.api_accuracy_checker.run_ut.run_ut_utils import get_validated_result_csv_path, \
    get_validated_details_csv_path
from msprobe.pytorch.api_accuracy_checker.compare.compare import Comparator
from msprobe.pytorch.common import parse_json_info_forward_backward
from msprobe.pytorch.common.log import logger
from msprobe.core.common.file_utils import FileChecker, check_file_suffix, check_link, FileOpen, \
    check_path_before_create, create_directory
from msprobe.core.common.file_utils import remove_path
from msprobe.core.common.const import FileCheckConst


def split_json_file(input_file, num_splits, filter_api):
    forward_data, backward_data, real_data_path = parse_json_info_forward_backward(input_file)
    if filter_api:
        forward_data = preprocess_forward_content(forward_data)
    for data_name in list(forward_data.keys()):
        forward_data[f"{data_name}.forward"] = forward_data.pop(data_name)
    for data_name in list(backward_data.keys()):
        backward_data[f"{data_name}.backward"] = backward_data.pop(data_name)

    with FileOpen(input_file, 'r') as file:
        input_data = json.load(file)
        input_data.pop("data")

    items = list(forward_data.items())
    total_items = len(items)
    chunk_size = total_items // num_splits
    split_files = []

    for i in range(num_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_splits - 1 else total_items

        split_forward_data = dict(items[start:end])
        temp_data = {
            **input_data,
            "data":{
                **split_forward_data,
                **backward_data
            }
        }
        split_filename = f"temp_part{i}.json"
        with FileOpen(split_filename, 'w') as split_file:
            json.dump(temp_data, split_file)
        split_files.append(split_filename)

    return split_files, total_items


def signal_handler(signum, frame):
    logger.warning(f'Signal handler called with signal {signum}')
    raise KeyboardInterrupt()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


ParallelUTConfig = namedtuple('ParallelUTConfig', ['api_files', 'out_path', 'num_splits',
                                                   'save_error_data_flag', 'jit_compile_flag', 'device_id',
                                                   'result_csv_path', 'total_items', 'config_path'])


def run_parallel_ut(config):
    processes = []
    device_id_cycle = cycle(config.device_id)
    if config.save_error_data_flag:
        logger.info("UT task error datas will be saved")
    logger.info(f"Starting parallel UT with {config.num_splits} processes")
    progress_bar = tqdm(total=config.total_items, desc="Total items", unit="items")

    def create_cmd(api_info, dev_id):
        dirname, filename = os.path.split(os.path.abspath(__file__))
        run_ut_path = os.path.join(dirname, "run_ut.py")
        cmd = [
            sys.executable, run_ut_path,
            '-api_info', api_info,
            *(['-o', config.out_path] if config.out_path else []),
            '-d', str(dev_id),
            *(['-j'] if config.jit_compile_flag else []),
            *(['-save_error_data'] if config.save_error_data_flag else []),
            '-csv_path', config.result_csv_path,
            *(['-config', config.config_path] if config.config_path else [])
        ]
        return cmd

    def read_process_output(process):
        try:
            while True:
                if process.poll() is not None:
                    break
                output = process.stdout.readline()
                if output == '':
                    break
                if '[ERROR]' in output:
                    logger.warning(output, end='')
                    sys.stdout.flush()
        except ValueError as e:
            logger.warning(f"An error occurred while reading subprocess output: {e}")

    def update_progress_bar(progress_bar, result_csv_path):
        while any(process.poll() is None for process in processes):
            with FileOpen(result_csv_path, 'r') as result_file:
                completed_items = len(result_file.readlines()) - 1
                progress_bar.update(completed_items - progress_bar.n)
            time.sleep(1)

    for api_info in config.api_files:
        cmd = create_cmd(api_info, next(device_id_cycle))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
                                   text=True, bufsize=1, shell=False)
        processes.append(process)
        threading.Thread(target=read_process_output, args=(process,), daemon=True).start()

    progress_bar_thread = threading.Thread(target=update_progress_bar, args=(progress_bar, config.result_csv_path))
    progress_bar_thread.start()

    def clean_up():
        progress_bar.close()
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
        for file in config.api_files:
            check_link(file)
            try:
                remove_path(file)
            except FileNotFoundError:
                logger.warning(f"File not found and could not be deleted: {file}")

    try:
        for process in processes:
            process.communicate(timeout=None)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, terminating processes and cleaning up...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        if progress_bar.n < config.total_items:
            logger.warning("The UT task has not been completed. The parameter '-csv_path' along with the path to " \
                           "the result CSV file will be utilized to resume the UT task.")
        clean_up()
        progress_bar_thread.join()
    try:
        comparator = Comparator(config.result_csv_path, config.result_csv_path, False)
        comparator.print_pretest_result()
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def prepare_config(args):
    check_link(args.api_info_file)
    api_info = os.path.realpath(args.api_info_file)
    check_file_suffix(api_info, FileCheckConst.JSON_SUFFIX)
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    check_path_before_create(out_path)
    create_directory(out_path)
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    split_files, total_items = split_json_file(api_info, args.num_splits, args.filter_api)
    config_path = os.path.realpath(args.config_path) if args.config_path else None
    result_csv_path = args.result_csv_path or os.path.join(
                      out_path, f"accuracy_checking_result_{time.strftime('%Y%m%d%H%M%S')}.csv")
    if not args.result_csv_path:
        details_csv_path = os.path.join(out_path, f"accuracy_checking_details_{time.strftime('%Y%m%d%H%M%S')}.csv")
        comparator = Comparator(result_csv_path, details_csv_path, False)
    else:
        result_csv_path = get_validated_result_csv_path(args.result_csv_path, 'result')
        details_csv_path = get_validated_details_csv_path(result_csv_path)
    logger.info(f"UT task result will be saved in {result_csv_path}")
    logger.info(f"UT task details will be saved in {details_csv_path}")
    return ParallelUTConfig(split_files, out_path, args.num_splits, args.save_error_data,
                            args.jit_compile, args.device_id, result_csv_path,
                            total_items, config_path)


def main():
    parser = argparse.ArgumentParser(description='Run UT in parallel')
    _run_ut_parser(parser)
    parser.add_argument('-n', '--num_splits', type=int, choices=range(1, 65), default=8, 
                        help='Number of splits for parallel processing. Range: 1-64')
    args = parser.parse_args()
    config = prepare_config(args)
    run_parallel_ut(config)


if __name__ == '__main__':
    main()
