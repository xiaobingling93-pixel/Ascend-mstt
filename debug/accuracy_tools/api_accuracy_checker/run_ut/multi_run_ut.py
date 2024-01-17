import subprocess
import json
import os
import sys
import argparse
import glob
import csv
from collections import namedtuple
from itertools import cycle
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileCheckConst, FileChecker, \
    check_file_suffix, check_link, FileOpen
from api_accuracy_checker.compare.compare import Comparator
from api_accuracy_checker.run_ut.run_ut import get_statistics_from_result_csv, _run_ut_parser
from api_accuracy_checker.dump.info_dump import write_json
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
        for item in items[start:end]:
            write_json(split_filename, {item[0]: item[1]})
        split_files.append(split_filename)

    return split_files


def merge_csv_files(csv_path, output_csv_path):
    if not csv_path:
        raise ValueError("No CSV paths provided for merging.")
    details_files = [path.replace('result', 'details') for path in csv_path]
    output_details_path = output_csv_path.replace('result', 'details')
    with open(csv_path[0], 'a', newline='') as merged_file:
        writer = csv.writer(merged_file)
        for csv_path in csv_path[1:]:
            with open(csv_path, 'r') as read_file:
                reader = csv.reader(read_file)
                header = next(reader)
                for row in reader:
                    writer.writerow(row)
            os.remove(csv_path)

    with open(details_files[0], 'a', newline='') as merged_file:
        writer = csv.writer(merged_file)
        for details_file in details_files[1:]:
            with open(details_file, 'r') as read_file:
                reader = csv.reader(read_file)
                header = next(reader)
                for row in reader:
                    writer.writerow(row)
            os.remove(details_file)

ParallelUTConfig = namedtuple('ParallelUTConfig', ['forward_files', 'backward_files', 'out_path', 'num_splits', 'save_error_data_flag', 'jit_compile_flag', 'device_ids', 'result_csv_path'])


def run_parallel_ut(config):
    processes = []
    device_id_cycle = cycle(config.device_ids)

    def create_cmd(fwd, bwd, dev_id):
        cmd = [
            sys.executable, 'run_ut.py',
            '-forward', fwd,
            '-backward' if bwd else '',
            bwd if bwd else '',
            '-o' if config.out_path else '', 
            config.out_path if config.out_path else '',
            '-d', str(dev_id),
            '-j' if config.jit_compile_flag else '',
            '-save_error_data' if config.save_error_data_flag else '',
            '-csv_path' if config.result_csv_path else '',
            config.result_csv_path if config.result_csv_path else ''
        ]
        return [arg for arg in cmd if arg]

    commands = [create_cmd(fwd, bwd, next(device_id_cycle)) for fwd, bwd in zip(config.forward_files, config.backward_files)]
    try:
        for cmd in commands:
            processes.append(subprocess.Popen(cmd))

        for process in processes:
            process.communicate()
    except KeyboardInterrupt: 
        print_error_log("Interrupted by user, terminating processes...")
        for process in processes:
            process.terminate()
            process.wait()

    for file in config.forward_files:
        os.remove(file)
    try:
        process_csv_and_print_results(config.out_path, config.result_csv_path)
    except FileNotFoundError as e:
        print_error_log(f"Error: {e}")
    except Exception as e:
        print_error_log(f"An unexpected error occurred: {e}")


def process_csv_and_print_results(out_path, result_csv_path=None):
    csv_files = glob.glob(os.path.join(out_path, 'accuracy_checking_result_*.csv'))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the specified output path.")
    latest_csv = max(csv_files, key=os.path.getmtime)

    comparator = Comparator(latest_csv, latest_csv, False)
    comparator.test_result_cnt = get_statistics_from_result_csv(latest_csv)
    comparator.print_pretest_result()


def main():
    parser = argparse.ArgumentParser(description='Run UT in parallel')
    _run_ut_parser(parser)
    parser.add_argument('-n', '--num_splits', type=int, choices=range(1, 65), default=64, help='Number of splits for parallel processing. Range: 1-64')
    parser.add_argument('-d', '--device_ids', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7], help='Device IDs for running tests. Default is 0 to 7.')
    parser.add_argument('-csv_path', '--result_csv_path', nargs='+', help='<optional> Paths of multiple accuracy_checking_result_{timestamp}.csv files to be merged.', required=False)
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
    if args.result_csv_path and len(args.result_csv_path) > 1:
        merge_csv_files(args.result_csv_path, args.result_csv_path[0])
        result_csv_path = args.result_csv_path[0]
    else:
        result_csv_path = args.result_csv_path if args.result_csv_path else None
    config = ParallelUTConfig(forward_splits, backward_splits, args.out_path, args.num_splits, args.save_error_data, args.jit_compile, args.device_ids, result_csv_path)
    run_parallel_ut(config)

if __name__ == '__main__':
    main()