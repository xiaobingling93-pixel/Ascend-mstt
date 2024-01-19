import subprocess
import json
import os
import sys
import argparse
import time
from collections import namedtuple
from itertools import cycle
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileCheckConst, FileChecker, \
    check_file_suffix, check_link, FileOpen
from api_accuracy_checker.compare.compare import Comparator
from api_accuracy_checker.run_ut.run_ut import _run_ut_parser
from api_accuracy_checker.common.utils import print_error_log


def split_json_file(input_file, num_splits):
    with FileOpen(input_file, 'r') as file:
        data = json.load(file)

    items = list(data.items())
    total_items = len(items)

    chunk_size = total_items // num_splits
    split_files = []

    for i in range(num_splits):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_splits - 1 else total_items
        split_filename = os.path.join("./", f"temp_part{i}.json")
        if os.path.exists(split_filename):
            os.remove(split_filename)
        with FileOpen(split_filename, 'w') as split_file:
            json.dump(dict(items[start:end]), split_file)
        split_files.append(split_filename)

    return split_files


ParallelUTConfig = namedtuple('ParallelUTConfig', ['forward_files', 'backward_files', 'out_path', 'num_splits', 'save_error_data_flag', 'jit_compile_flag', 'device_id', 'result_csv_path'])


def run_parallel_ut(config):
    processes = []
    device_id_cycle = cycle(config.device_id)

    def create_cmd(fwd, bwd, dev_id):
        cmd = [
            sys.executable, 'run_ut.py',
            '-forward', fwd,
            *(['-backward', bwd] if bwd else []),
            *(['-o', config.out_path] if config.out_path else []),
            '-d', str(dev_id),
            *(['-j'] if config.jit_compile_flag else []),
            *(['-save_error_data'] if config.save_error_data_flag else []),
            *(['-csv_path', config.result_csv_path] if config.result_csv_path else [])
        ]
        return cmd

    for fwd, bwd in zip(config.forward_files, config.backward_files):
        cmd = create_cmd(fwd, bwd, next(device_id_cycle))
        processes.append(subprocess.Popen(cmd))

    try:
        for process in processes:
            process.communicate(timeout=None)
    except KeyboardInterrupt: 
        print_error_log("Interrupted by user, terminating processes...")
        for process in processes:
            process.terminate()
            process.wait()
    finally:
        for file in config.forward_files:
            if os.path.exists(file):
                os.remove(file)


def main():
    parser = argparse.ArgumentParser(description='Run UT in parallel')
    _run_ut_parser(parser)
    parser.add_argument('-n', '--num_splits', type=int, choices=range(1, 65), default=8, help='Number of splits for parallel processing. Range: 1-64')
    args = parser.parse_args()
    check_link(args.forward_input_file)
    check_link(args.backward_input_file)
    forward_file = os.path.realpath(args.forward_input_file)
    backward_file = os.path.realpath(args.backward_input_file) if args.backward_input_file else None
    check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    forward_splits = split_json_file(args.forward_input_file, args.num_splits)
    backward_splits = [backward_file] * args.num_splits if backward_file else [None] * args.num_splits
    if not args.result_csv_path:
        current_time = time.strftime("%Y%m%d%H%M%S")
        result_csv_path = os.path.join(out_path, "accuracy_checking_result_{}.csv".format(current_time))
        details_csv_path = os.path.join(out_path, "accuracy_checking_details_{}.csv".format(current_time))
        comparator = Comparator(result_csv_path, details_csv_path, False)
        args.result_csv_path = result_csv_path
    else:
        result_csv_path = args.result_csv_path
    config = ParallelUTConfig(forward_splits, backward_splits, args.out_path, args.num_splits, args.save_error_data, args.jit_compile, args.device_id, result_csv_path)
    run_parallel_ut(config)

if __name__ == '__main__':
    main()