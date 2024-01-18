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
from api_accuracy_checker.run_ut.run_ut import get_statistics_from_result_csv
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
        with FileOpen(split_filename, 'w') as split_file:
            json.dump(dict(items[start:end]), split_file)
        split_files.append(split_filename)

    return split_files


def merge_csv_files(csv_paths):
    if not csv_paths:
        raise ValueError("No CSV paths provided for merging.")
    details_files = [path.replace('result', 'details') for path in csv_paths]
    for csv_path, details_path in zip(csv_paths, details_files):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"The CSV file {csv_path} does not exist.")
        if not os.path.exists(details_path):
            raise FileNotFoundError(f"The details file {details_path} does not exist.")
        if not csv_path.replace('result', 'details') == details_path:
            raise ValueError(f"The CSV file {csv_path} does not match the details file {details_path}.")

    def merge_files(file_paths, output_file):
        with FileOpen(output_file, 'a') as merged_file:
            writer = csv.writer(merged_file)
            for file_path in file_paths[1:]:
                with FileOpen(file_path, 'r') as read_file:
                    reader = csv.reader(read_file)
                    next(reader)
                    writer.writerows(reader)
                os.remove(file_path)

    merge_files(csv_paths, csv_paths[0])
    merge_files(details_files, details_files[0])

ParallelUTConfig = namedtuple('ParallelUTConfig', ['forward_files', 'backward_files', 'out_path', 'num_splits', 'save_error_data_flag', 'jit_compile_flag', 'device', 'result_csv_path'])


def run_parallel_ut(config):
    processes = []
    device_id_cycle = cycle(config.device)

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
    parser.add_argument("-forward", "--forward_input_file", dest="forward_input_file", default="", type=str,
                        help="<Required> The api param tool forward result file: generate from api param tool, "
                             "a json file.",
                        required=True)
    parser.add_argument("-backward", "--backward_input_file", dest="backward_input_file", default="", type=str,
                        help="<Required> The api param tool backward result file: generate from api param tool, "
                             "a json file.",
                        required=False)
    parser.add_argument("-o", "--out_path", dest="out_path", default="", type=str,
                        help="<optional> The ut task result out path.",
                        required=False)
    parser.add_argument('-save_error_data', dest="save_error_data", action="store_true",
                        help="<optional> Save compare failed api output.", required=False)
    parser.add_argument("-j", "--jit_compile", dest="jit_compile", action="store_true",
                        help="<optional> whether to turn on jit compile", required=False)
    parser.add_argument('-n', '--num_splits', type=int, choices=range(1, 65), default=8, help='Number of splits for parallel processing. Range: 1-64')
    parser.add_argument('-d', '--device', nargs='+', type=int, default=[0], help='Device IDs for running tests. Default is 0.')
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
        merge_csv_files(args.result_csv_path)
        result_csv_path = args.result_csv_path[0]
    else:
        result_csv_path = args.result_csv_path if args.result_csv_path else None
    config = ParallelUTConfig(forward_splits, backward_splits, args.out_path, args.num_splits, args.save_error_data, args.jit_compile, args.device, result_csv_path)
    run_parallel_ut(config)

if __name__ == '__main__':
    main()