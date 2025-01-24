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

import argparse
import os
from collections import namedtuple

from msprobe.core.common.file_utils import create_directory
from msprobe.pytorch.parse_tool.lib.compare import Compare
from msprobe.pytorch.parse_tool.lib.config import Const
from msprobe.pytorch.parse_tool.lib.parse_exception import catch_exception, ParseException
from msprobe.pytorch.parse_tool.lib.utils import Util
from msprobe.pytorch.parse_tool.lib.visualization import Visualization


class ParseTool:
    def __init__(self):
        self.util = Util()
        self.compare = Compare()
        self.visual = Visualization()

    @catch_exception
    def prepare(self):
        create_directory(Const.DATA_ROOT_DIR)

    @catch_exception
    def do_vector_compare(self, args):
        if not args.output_path:
            result_dir = os.path.join(Const.COMPARE_DIR)
        else:
            result_dir = args.output_path
        my_dump_path = args.my_dump_path
        golden_dump_path = args.golden_dump_path
        if not os.path.isdir(my_dump_path) or not os.path.isdir(golden_dump_path):
            self.util.log.error("Please enter a directory not a file")
            raise ParseException(ParseException.PARSE_INVALID_PATH_ERROR)
        msaccucmp_path = self.util.path_strip(args.msaccucmp_path) if args.msaccucmp_path else Const.MS_ACCU_CMP_PATH
        self.util.check_path_valid(msaccucmp_path)
        self.util.check_executable_file(msaccucmp_path)
        self.compare.npu_vs_npu_compare(my_dump_path, golden_dump_path, result_dir, msaccucmp_path)

    @catch_exception
    def do_convert_dump(self, argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-n', '--name', dest='path', default=None, required=True, help='dump file or dump file directory')
        parser.add_argument(
            '-f', '--format', dest='format', default=None, required=False, help='target format')
        parser.add_argument(
            '-out', '--output_path', dest='output_path', required=False, default=None, help='output path')
        parser.add_argument(
            "-cmp_path", "--msaccucmp_path", dest="msaccucmp_path", default=None,
            help="<Optional> the msaccucmp.py file path", required=False)
        args = parser.parse_args(argv)
        self.util.check_path_valid(args.path)
        self.util.check_files_in_path(args.path)
        msaccucmp_path = self.util.path_strip(args.msaccucmp_path) if args.msaccucmp_path else Const.MS_ACCU_CMP_PATH
        self.util.check_path_valid(msaccucmp_path)
        self.util.check_executable_file(msaccucmp_path)
        if args.format:
            self.util.check_str_param(args.format)
        self.compare.convert_dump_to_npy(args.path, args.format, args.output_path, msaccucmp_path)

    @catch_exception
    def do_print_data(self, argv=None):
        """print tensor data"""
        parser = argparse.ArgumentParser()
        parser.add_argument('-n', '--name', dest='path', default=None, required=True, help='File name')
        args = parser.parse_args(argv)
        self.visual.print_npy_data(args.path)

    @catch_exception
    def do_parse_pkl(self, argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-f', '--file', dest='file_name', default=None, required=True, help='PKL file path')
        parser.add_argument(
            '-n', '--name', dest='api_name', default=None, required=True, help='API name')
        args = parser.parse_args(argv)
        self.visual.parse_pkl(args.file_name, args.api_name)

    @catch_exception
    def do_compare_data(self, argv):
        """compare two tensor"""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m", "--my_dump_path", dest="my_dump_path", default=None,
            help="<Required> my dump path, the data compared with golden data",
            required=True
        )
        parser.add_argument(
            "-g", "--golden_dump_path", dest="golden_dump_path", default=None,
            help="<Required> the golden dump data path",
            required=True
        )
        parser.add_argument('-p', '--print', dest='count', default=20, type=int, help='print err data num')
        parser.add_argument('-s', '--save', dest='save', action='store_true', help='save data in txt format')
        parser.add_argument('-al', '--atol', dest='atol', default=0.001, type=float, help='set rtol')
        parser.add_argument('-rl', '--rtol', dest='rtol', default=0.001, type=float, help='set atol')
        args = parser.parse_args(argv)
        self.util.check_positive(args.count)
        self.util.check_positive(args.rtol)
        self.util.check_positive(args.atol)
        self.util.check_path_valid(args.my_dump_path)
        self.util.check_path_valid(args.golden_dump_path)
        self.util.check_file_path_format(args.my_dump_path, Const.NPY_SUFFIX)
        self.util.check_file_path_format(args.golden_dump_path, Const.NPY_SUFFIX)
        compare_data_args = namedtuple('compare_data_args',
                                       ['my_dump_path', 'golden_dump_path', 'save', 'rtol', 'atol', 'count'])
        compare_data_args.__new__.__defaults__ = (False, 0.001, 0.001, 20)
        res = compare_data_args(args.my_dump_path, args.golden_dump_path, args.save, args.rtol, args.atol, args.count)
        self.compare.compare_data(res)

    @catch_exception
    def do_compare_converted_dir(self, args):
        """compare two dir"""
        my_dump_dir = self.util.path_strip(args.my_dump_path)
        golden_dump_dir = self.util.path_strip(args.golden_dump_path)
        if my_dump_dir == golden_dump_dir:
            self.util.log.error("My directory path and golden directory path is same. Please check parameter"
                                " '-m' and '-g'.")
            raise ParseException("My directory path and golden directory path is same.")
        output_path = self.util.path_strip(args.output_path) if args.output_path else Const.BATCH_COMPARE_DIR
        create_directory(output_path)
        self.compare.compare_converted_dir(my_dump_dir, golden_dump_dir, output_path)

    @catch_exception
    def do_convert_api_dir(self, argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m", "--my_dump_path", dest="my_dump_path", default=None,
            help="<Required> my dump path, the data need to convert to npy files.",
            required=True
        )
        parser.add_argument(
            '-out', '--output_path', dest='output_path', required=False, default=None, help='output path')
        parser.add_argument(
            "-asc", "--msaccucmp_path", dest="msaccucmp_path", default=None,
            help="<Optional> the msaccucmp.py file path", required=False)
        args = parser.parse_args(argv)
        self.util.check_path_valid(args.my_dump_path)
        self.util.check_files_in_path(args.my_dump_path)
        output_path = self.util.path_strip(args.output_path) if args.output_path else \
            os.path.join(Const.BATCH_DUMP_CONVERT_DIR, self.util.localtime_str())
        msaccucmp_path = self.util.path_strip(
            args.msaccucmp_path) if args.msaccucmp_path else Const.MS_ACCU_CMP_PATH
        self.util.check_path_valid(msaccucmp_path)
        self.util.check_executable_file(msaccucmp_path)
        self.compare.convert_api_dir_to_npy(args.my_dump_path, None, output_path, msaccucmp_path)
