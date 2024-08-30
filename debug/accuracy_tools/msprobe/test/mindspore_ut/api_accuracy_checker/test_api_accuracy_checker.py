import unittest
import sys
import logging
import os
import json
import csv

from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import ApiAccuracyChecker

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)

def delete_files_with_prefix(directory, prefix):
    for file_or_folder in os.listdir(directory):
        full_path = os.path.join(directory, file_or_folder)
        if os.path.isfile(full_path) and file_or_folder.startswith(prefix):
            os.remove(full_path)
            print(f"已删除文件: {full_path}")

def modify_tensor_api_info_json(json_file_path, modified_dump_data_dir):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    if 'dump_data_dir' in data:
        data['dump_data_dir'] = modified_dump_data_dir
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def check_csv(csv_path, row_num):
    with open(csv_path, 'r', encoding="utf-8") as f:
        csvreader = csv.reader(f)
        assert row_num == len(csvreader)

def find_with_prefix(directory, prefix):
    entries = os.listdir(directory)
    target_files = [entry for entry in entries if entry.startswith(prefix) and os.path.isfile(os.path.join(directory, entry))]
    return target_files

class TestApiAccuracyChecker(unittest.TestCase):

    def test_statistics_mode(self):
        api_info_statistics_path = os.path.join(directory, "files", "api_info_statistics.json")
        result_directory = os.path.join(directory, "files")
        api_accuracy_checker = ApiAccuracyChecker()
        api_accuracy_checker.parse(api_info_statistics_path)
        api_accuracy_checker.run_and_compare()
        api_accuracy_checker.to_detail_csv(result_directory)
        api_accuracy_checker.to_result_csv(result_directory)
        detail_csv = find_with_prefix(result_directory, "accuracy_checking_detail")
        assert len(detail_csv) == 1
        check_csv(detail_csv[0], 3)

        result_csv = find_with_prefix(result_directory, "accuracy_checking_result")
        assert len(result_csv) == 1
        check_csv(result_csv[0], 2)
        delete_files_with_prefix(result_directory, "accuracy_checking")

    def test_tensor_mode(self):
        api_info_tensor_path = os.path.join(directory, "files", "api_info_tensor.json")
        result_directory = os.path.join(directory, "files")
        modify_tensor_api_info_json(api_info_tensor_path, result_directory)
        api_accuracy_checker = ApiAccuracyChecker()
        api_accuracy_checker.parse(api_info_tensor_path)
        api_accuracy_checker.run_and_compare()
        api_accuracy_checker.to_detail_csv(result_directory)
        api_accuracy_checker.to_result_csv(result_directory)
        detail_csv = find_with_prefix(result_directory, "accuracy_checking_detail")
        assert len(detail_csv) == 1
        check_csv(detail_csv[0], 3)

        result_csv = find_with_prefix(result_directory, "accuracy_checking_result")
        assert len(result_csv) == 1
        check_csv(result_csv[0], 2)
        modify_tensor_api_info_json(api_info_tensor_path, "")
        delete_files_with_prefix(result_directory, "accuracy_checking")

    def test_index_select_api_statistics_mode(self):
        api_info_statistics_path = os.path.join(directory, "files", "index_select_statistics_dump.json")
        result_directory = os.path.join(directory, "files")
        api_accuracy_checker = ApiAccuracyChecker()
        api_accuracy_checker.parse(api_info_statistics_path)
        api_accuracy_checker.run_and_compare()
        api_accuracy_checker.to_detail_csv(result_directory)
        api_accuracy_checker.to_result_csv(result_directory)
        detail_csv = find_with_prefix(result_directory, "accuracy_checking_detail")
        assert len(detail_csv) == 1
        check_csv(detail_csv[0], 3)

        result_csv = find_with_prefix(result_directory, "accuracy_checking_result")
        assert len(result_csv) == 1
        check_csv(result_csv[0], 2)
        delete_files_with_prefix(result_directory, "accuracy_checking")

    def test_index_select_api_tensor_mode(self):
        api_info_tensor_path = os.path.join(directory, "files", "index_select_tensor_dump.json")
        result_directory = os.path.join(directory, "files")
        modify_tensor_api_info_json(api_info_tensor_path, result_directory)
        api_accuracy_checker = ApiAccuracyChecker()
        api_accuracy_checker.parse(api_info_tensor_path)
        api_accuracy_checker.run_and_compare()
        api_accuracy_checker.to_detail_csv(result_directory)
        api_accuracy_checker.to_result_csv(result_directory)
        detail_csv = find_with_prefix(result_directory, "accuracy_checking_detail")
        assert len(detail_csv) == 1
        check_csv(detail_csv[0], 3)

        result_csv = find_with_prefix(result_directory, "accuracy_checking_result")
        assert len(result_csv) == 1
        check_csv(result_csv[0], 2)
        modify_tensor_api_info_json(api_info_tensor_path, "")
        delete_files_with_prefix(result_directory, "accuracy_checking")

if __name__ == '__main__':
    unittest.main()