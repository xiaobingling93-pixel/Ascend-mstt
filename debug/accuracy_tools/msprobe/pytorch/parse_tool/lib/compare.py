# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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
import time
from collections import namedtuple

import numpy as np

from msprobe.core.common.file_utils import create_directory, load_npy, save_npy_to_txt, write_csv, os_walk_for_files
from msprobe.pytorch.parse_tool.lib.config import Const
from msprobe.pytorch.parse_tool.lib.parse_exception import ParseException
from msprobe.pytorch.parse_tool.lib.utils import Util


class Compare:
    def __init__(self):
        self.util = Util()
        self.log = self.util.log
        self.vector_compare_result = {}

    def npu_vs_npu_compare(self, my_dump_path, golden_dump_path, result_dir, msaccucmp_path):
        self.log.info("Start Compare ...............")
        self.compare_vector(my_dump_path, golden_dump_path, result_dir, msaccucmp_path)
        self.log.info("Compare finished!!")

    def compare_vector(self, my_dump_path, golden_dump_path, result_dir, msaccucmp_path):
        create_directory(result_dir)
        self.util.check_path_valid(result_dir)
        call_msaccucmp = self.util.check_msaccucmp(msaccucmp_path)
        cmd = '%s %s compare -m %s -g %s -out %s' % (
            self.util.python, call_msaccucmp, my_dump_path, golden_dump_path, result_dir
        )
        return self.util.execute_command(cmd)

    def convert_dump_to_npy(self, dump_file, data_format, output, msaccucmp_path):
        dump_file = self.util.path_strip(dump_file)
        file_name = ""
        if os.path.isfile(dump_file):
            self.log.info("Covert file is: %s" % dump_file)
            file_name = os.path.basename(dump_file)
        elif os.path.isdir(dump_file):
            self.log.info("Convert all files in path: %s" % dump_file)
            file_name = ""
        output = output if output else Const.DUMP_CONVERT_DIR
        convert = self.convert(dump_file, data_format, output, msaccucmp_path)
        if convert == 0:
            convert_files = self.util.list_convert_files(output, file_name)

            summary_txt = ["SrcFile: %s" % dump_file]
            for convert_file in convert_files.values():
                summary_txt.append(" - %s" % convert_file.file_name)
            self.log.info("Transfer result is saved in : %s" % os.path.realpath(output))
            self.util.print_panel("\n".join(summary_txt))

    def convert(self, dump_file, data_format, output, msaccucmp_path):
        create_directory(output)
        self.util.check_path_valid(output)
        call_msaccucmp = self.util.check_msaccucmp(msaccucmp_path)
        if data_format:
            cmd = '%s %s convert -d %s -out %s -f %s' % (
                self.util.python, call_msaccucmp, dump_file, output, data_format
            )
        else:
            cmd = '%s %s convert -d %s -out %s' % (
                self.util.python, call_msaccucmp, dump_file, output
            )
        return self.util.execute_command(cmd)

    def compare_data(self, args):
        """Compare data"""
        (left, right, save_txt, rl, al, diff_count) = args
        if left is None or right is None:
            raise ParseException("invalid input or output")
        if self.util.check_path_valid(left) and self.util.check_path_valid(right):
            left_data = load_npy(left)
            right_data = load_npy(right)
        # save to txt
        if save_txt:
            save_npy_to_txt(left_data, left + ".txt")
            save_npy_to_txt(right_data, right + ".txt")
        # compare data
        (total_cnt, all_close, cos_sim, err_percent) = self.do_compare_data(left_data, right_data, rl, al, diff_count)
        content = ['Left:', ' ├─ NpyFile: %s' % left]
        if save_txt:
            content.append(' ├─ TxtFile: [green]%s.txt[/green]' % left)
        content.append(' └─ NpySpec: [yellow]%s[/yellow]' % self.util.gen_npy_info_txt(left_data))
        content.append('Right:')
        content.append(' ├─ NpyFile: %s' % right)
        if save_txt:
            content.append(' ├─ TxtFile: [green]%s.txt[/green]' % right)
        content.append(' └─ NpySpec: [yellow]%s[/yellow]' % self.util.gen_npy_info_txt(right_data))
        content.append('NumCnt:   %s' % total_cnt)
        content.append('AllClose: %s' % all_close)
        content.append('CosSim:   %s' % cos_sim)
        content.append('ErrorPer: %s  (rl= %s, al= %s)' % (err_percent, rl, al))
        self.util.print_panel("\n".join(content))

    def do_compare_data(self, left, right, rl=0.001, al=0.001, diff_count=20):
        data_left = left.astype(np.float32)
        data_right = right.astype(np.float32)
        shape_left = data_left.shape
        shape_right = data_right.shape
        if shape_left != shape_right:
            self.log.warning("Data shape not equal: %s vs %s" % (data_left.shape, data_right.shape))
        data_left = data_left.reshape(-1)
        data_right = data_right.reshape(-1)
        if data_left.shape[0] != data_right.shape[0]:
            self.log.warning("Data size not equal: %s vs %s" % (data_left.shape, data_right.shape))
            if data_left.shape[0] < data_right.shape[0]:
                data_left = np.pad(data_left, (0, data_right.shape[0] - data_left.shape[0]), 'constant')
            else:
                data_right = np.pad(data_right, (0, data_left.shape[0] - data_right.shape[0]), 'constant')
        all_close = np.allclose(data_left, data_right, atol=al, rtol=rl)
        np.seterr(divide='raise')
        cos_sim = np.dot(data_left, data_right) / (
                np.sqrt(np.dot(data_left, data_left)) * np.sqrt(np.dot(data_right, data_right)))
        err_cnt = 0
        total_cnt = data_left.shape[0]
        diff_table_columns = ['Index', 'Left', 'Right', 'Diff']
        err_table = self.util.create_table("Error Item Table", diff_table_columns)
        top_table = self.util.create_table("Top Item Table", diff_table_columns)
        for i in range(total_cnt):
            abs_diff = abs(data_left[i] - data_right[i])
            if i < diff_count:
                top_table.add_row(str(i), str(data_left[i]), str(data_right[i]), str(abs_diff))
            if abs_diff > (al + rl * abs(data_right[i])):
                if err_cnt < diff_count:
                    err_table.add_row(str(i), str(data_left[i]), str(data_right[i]), str(abs_diff))
                err_cnt += 1
        if total_cnt == 0:
            err_percent = float(0)
        else:
            err_percent = float(err_cnt / total_cnt)
        self.util.print(self.util.create_columns([err_table, top_table]))
        do_compare_data_result = namedtuple('do_compare_data_result', ['cnt', 'close', 'cos', 'err'])
        res = do_compare_data_result(total_cnt, all_close, cos_sim, err_percent)
        return res

    def compare_npy(self, file, bench_file, output_path):
        if self.util.check_path_valid(file) and self.util.check_path_valid(bench_file):
            data = load_npy(file)
            bench_data = load_npy(bench_file)
        shape, dtype = data.shape, data.dtype
        bench_shape, bench_dtype = bench_data.shape, bench_data.dtype
        filename = os.path.basename(file)
        bench_filename = os.path.basename(bench_file)
        if shape != bench_shape or dtype != bench_dtype:
            self.log.error(
                "Shape or dtype between two npy files is inconsistent. Please check the two files."
                "File 1: %s, file 2: %s" % (file, bench_file))
            self.util.deal_with_dir_or_file_inconsistency(output_path)
            return
        md5_consistency = False
        if self.util.get_md5_for_numpy(data) == self.util.get_md5_for_numpy(bench_data):
            md5_consistency = True
        data_mean = np.mean(data)
        bench_data_mean = np.mean(bench_data)
        abs_error = np.abs(data - bench_data)
        bench_data = self.util.deal_with_value_if_has_zero(bench_data)
        rel_error = np.abs(abs_error / bench_data)
        abs_diff_max = abs_error.max()
        rel_diff_max = np.max(rel_error)
        compare_result = [[filename, bench_filename, data_mean, bench_data_mean, md5_consistency, abs_diff_max,
                           rel_diff_max]]
        write_csv(compare_result, output_path)

    def compare_all_file_in_directory(self, my_dump_dir, golden_dump_dir, output_path):
        if not (self.util.is_subdir_count_equal(my_dump_dir, golden_dump_dir)
                and self.util.check_npy_files_valid_in_dir(my_dump_dir)
                and self.util.check_npy_files_valid_in_dir(golden_dump_dir)):
            self.log.error(
                "Top level(Npy files level) directory structure is inconsistent. Please check the two directory.")
            self.util.deal_with_dir_or_file_inconsistency(output_path)
            return
        my_npy_files = self.util.get_sorted_files_names(my_dump_dir)
        golden_npy_files = self.util.get_sorted_files_names(golden_dump_dir)
        for my_npy_file_name, golden_npy_file_name in zip(my_npy_files, golden_npy_files):
            my_npy_path = os.path.join(my_dump_dir, my_npy_file_name)
            golden_npy_path = os.path.join(golden_dump_dir, golden_npy_file_name)
            self.compare_npy(my_npy_path, golden_npy_path, output_path)

    def compare_timestamp_directory(self, my_dump_dir, golden_dump_dir, output_path):
        if not self.util.is_subdir_count_equal(my_dump_dir, golden_dump_dir):
            self.log.error(
                "Second level(Timestamp level) directory structure is inconsistent. Please check the two directory.")
            self.util.deal_with_dir_or_file_inconsistency(output_path)
            return
        my_ordered_subdirs = self.util.get_sorted_subdirectories_names(my_dump_dir)
        golden_ordered_subdirs = self.util.get_sorted_subdirectories_names(golden_dump_dir)
        for my_subdir_name, golden_subdir_name in zip(my_ordered_subdirs, golden_ordered_subdirs):
            my_subdir_path = os.path.join(my_dump_dir, my_subdir_name)
            golden_subdir_path = os.path.join(golden_dump_dir, golden_subdir_name)
            self.compare_all_file_in_directory(my_subdir_path, golden_subdir_path, output_path)

    def compare_converted_dir(self, my_dump_dir, golden_dump_dir, output_dir):
        if not self.util.is_subdir_count_equal(my_dump_dir, golden_dump_dir):
            self.log.error(
                "Top level(Opname level) directory structure is inconsistent. Please check the two directory.")
            return
        timestamp = int(time.time())
        output_file_name = f"batch_compare_{timestamp}.csv"
        output_path = os.path.join(output_dir, output_file_name)
        title_rows = [[
            "NPU File Name",
            "Bench File Name",
            "Mean",
            "Bench Mean",
            "Md5 Consistency",
            "Max Abs Error",
            "Max Relative Error"
        ]]
        write_csv(title_rows, output_path)

        my_ordered_subdirs = self.util.get_sorted_subdirectories_names(my_dump_dir)
        golden_ordered_subdirs = self.util.get_sorted_subdirectories_names(golden_dump_dir)
        for my_subdir_name, golden_subdir_name in zip(my_ordered_subdirs, golden_ordered_subdirs):
            if not my_subdir_name == golden_subdir_name:
                self.log.error(
                    "Top level(Opname level) directory structure is inconsistent. Please check the two directory.")
                self.util.deal_with_dir_or_file_inconsistency(output_path)
                return
            my_subdir_path = os.path.join(my_dump_dir, my_subdir_name)
            golden_subdir_path = os.path.join(golden_dump_dir, golden_subdir_name)
            self.compare_timestamp_directory(my_subdir_path, golden_subdir_path, output_path)
        self.util.change_filemode_safe(output_path)
        self.log.info("Compare result is saved in : %s" % (output_path))

    def convert_api_dir_to_npy(self, dump_dir, param, output_dir, msaccucmp_path):
        dump_dir = self.util.path_strip(dump_dir)
        files = os_walk_for_files(dump_dir, Const.MAX_TRAVERSAL_DEPTH)
        filepaths = [os.path.join(file['root'], file['file']) for file in files]
        for path in filepaths:
            filename = os.path.basename(path)
            parts = filename.split(".")
            if len(parts) < 5:
                continue
            op_name = parts[1]
            timestamp = parts[-1]
            output_path = os.path.join(output_dir, op_name, timestamp)
            self.convert_dump_to_npy(path, param, output_path, msaccucmp_path)
