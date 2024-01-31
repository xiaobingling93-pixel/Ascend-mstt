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
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileCheckConst, FileChecker, \
    check_file_suffix, check_link, FileOpen
from api_accuracy_checker.compare.compare import Comparator
from api_accuracy_checker.run_ut.run_ut import _run_ut_parser, get_validated_result_csv_path, get_validated_details_csv_path
from api_accuracy_checker.common.utils import print_error_log, print_warn_log, print_info_log


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
        split_filename = f"temp_part{i}.json"
        with FileOpen(split_filename, 'w') as split_file:
            json.dump(dict(items[start:end]), split_file)
        split_files.append(split_filename)

    return split_files, total_items


def signal_handler(signum, frame):
    print_warn_log(f'Signal handler called with signal {signum}')
    raise KeyboardInterrupt()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


ParallelUTConfig = namedtuple('ParallelUTConfig', ['forward_files', 'backward_files', 'out_path', 'num_splits', 'save_error_data_flag', 'jit_compile_flag', 'device_id', 'result_csv_path', 'total_items', 'real_data_path'])


def run_parallel_ut(config):
    processes = []
    device_id_cycle = cycle(config.device_id)
    if config.save_error_data_flag:
        print_info_log("UT task error datas will be saved")
    print_info_log(f"Starting parallel UT with {config.num_splits} processes")
    progress_bar = tqdm(total=config.total_items, desc="Total items", unit="items")

    def create_cmd(fwd, bwd, dev_id):
        cmd = [
            sys.executable, 'run_ut.py',
            '-forward', fwd,
            *(['-backward', bwd] if bwd else []),
            *(['-o', config.out_path] if config.out_path else []),
            '-d', str(dev_id),
            *(['-j'] if config.jit_compile_flag else []),
            *(['-save_error_data'] if config.save_error_data_flag else []),
            '-csv_path', config.result_csv_path,
            *(['-real_data_path', config.real_data_path] if config.real_data_path else [])
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
                print(output, end='')
    except ValueError as e:
        print_warn_log(f"An error occurred while reading subprocess output: {e}")
    
    def update_progress_bar(progress_bar, result_csv_path):
        while any(process.poll() is None for process in processes):
            try:
                with open(result_csv_path, 'r') as result_file:
                    completed_items = len(result_file.readlines()) - 1
                    progress_bar.update(completed_items - progress_bar.n)
            except FileNotFoundError:
                print_warn_log(f"Result CSV file not found: {result_csv_path}.")
            except Exception as e:
                print_error_log(f"An unexpected error occurred while reading result CSV: {e}")
            time.sleep(10)
    
    for fwd, bwd in zip(config.forward_files, config.backward_files):
        cmd = create_cmd(fwd, bwd, next(device_id_cycle))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        processes.append(process)
        threading.Thread(target=read_process_output, args=(process,), daemon=True).start()

    progress_bar_thread = threading.Thread(target=update_progress_bar, args=(progress_bar, config.result_csv_path))
    progress_bar_thread.start()

    def clean_up():
        progress_bar.close()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for file in config.forward_files:
            try:
                os.remove(file)
            except FileNotFoundError:
                print_warn_log(f"File not found and could not be deleted: {file}")

    try:
        for process in processes:
            process.communicate(timeout=None)
    except KeyboardInterrupt: 
        print_warn_log("Interrupted by user, terminating processes and cleaning up...")
    except Exception as e:
        print_error_log(f"An unexpected error occurred: {e}")
    finally:
        clean_up()
        progress_bar_thread.join()
    try:
        comparator = Comparator(config.result_csv_path, config.result_csv_path, False)
        comparator.print_pretest_result()
    except FileNotFoundError as e:
        print_error_log(f"Error: {e}")
    except Exception as e:
        print_error_log(f"An unexpected error occurred: {e}")


def prepare_config(args):
    check_link(args.forward_input_file)
    check_link(args.backward_input_file) if args.backward_input_file else None
    forward_file = os.path.realpath(args.forward_input_file)
    backward_file = os.path.realpath(args.backward_input_file) if args.backward_input_file else None
    check_file_suffix(forward_file, FileCheckConst.JSON_SUFFIX)
    out_path = os.path.realpath(args.out_path) if args.out_path else "./"
    out_path_checker = FileChecker(out_path, FileCheckConst.DIR, ability=FileCheckConst.WRITE_ABLE)
    out_path = out_path_checker.common_check()
    forward_splits, total_items = split_json_file(args.forward_input_file, args.num_splits)
    backward_splits = [backward_file] * args.num_splits if backward_file else [None] * args.num_splits
    result_csv_path = args.result_csv_path or os.path.join(out_path, f"accuracy_checking_result_{time.strftime('%Y%m%d%H%M%S')}.csv")
    if not args.result_csv_path:
        details_csv_path = os.path.join(out_path, f"accuracy_checking_details_{time.strftime('%Y%m%d%H%M%S')}.csv")
        comparator = Comparator(result_csv_path, details_csv_path, False)
        print_info_log(f"UT task result will be saved in {result_csv_path}")
        print_info_log(f"UT task details will be saved in {details_csv_path}")
    else:
        result_csv_path = get_validated_result_csv_path(args.result_csv_path)
        details_csv_path = get_validated_details_csv_path(result_csv_path)
        print_info_log(f"UT task result will be saved in {result_csv_path}")
        print_info_log(f"UT task details will be saved in {details_csv_path}")
    return ParallelUTConfig(forward_splits, backward_splits, out_path, args.num_splits, args.save_error_data, args.jit_compile, args.device_id, result_csv_path, total_items, args.real_data_path)


def main():
    parser = argparse.ArgumentParser(description='Run UT in parallel')
    _run_ut_parser(parser)
    parser.add_argument('-n', '--num_splits', type=int, choices=range(1, 65), default=8, help='Number of splits for parallel processing. Range: 1-64')
    args = parser.parse_args()
    config = prepare_config(args)
    run_parallel_ut(config)

if __name__ == '__main__':
    main()
