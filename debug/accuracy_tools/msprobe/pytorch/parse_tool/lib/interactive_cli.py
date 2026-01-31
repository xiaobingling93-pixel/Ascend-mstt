# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


import argparse
import cmd

from msprobe.pytorch.parse_tool.lib.config import Const
from msprobe.pytorch.parse_tool.lib.parse_exception import catch_exception
from msprobe.pytorch.parse_tool.lib.parse_tool import ParseTool
from msprobe.pytorch.parse_tool.lib.utils import Util


class InteractiveCli(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.prompt = "Parse >>> "
        self.parse_tool = ParseTool()
        self.util = Util()
        self.util.print_panel(Const.HEADER)
        self.prepare()

    @staticmethod
    def _parse_argv(line, insert=None):
        argv = line.split() if line != "" else []
        if "-h" in argv:
            return argv
        if insert is not None and len(argv) and argv[0] != insert:
            argv.insert(0, insert)
        return argv

    def prepare(self):
        self.parse_tool.prepare()

    @catch_exception
    def default(self, line=""):
        self.stdout.write("Command invalid, Only support command start with cad/vc/dc/pk/cn/pt\n")

    @catch_exception
    def do_vc(self, line=""):
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
        parser.add_argument(
            "-out", "--output_path", dest="output_path", default=None,
            help="<Optional> the output path",
            required=False
        )
        parser.add_argument(
            "-cmp_path", "--msaccucmp_path", dest="msaccucmp_path", default=None,
            help="<Optional> the msaccucmp.py file path",
            required=False
        )
        args = parser.parse_args(self._parse_argv(line))
        self.util.check_path_valid(args.my_dump_path)
        self.util.check_path_valid(args.golden_dump_path)
        self.util.check_files_in_path(args.my_dump_path)
        self.util.check_files_in_path(args.golden_dump_path)
        if self.util.dir_contains_only(args.my_dump_path, ".npy") and \
                self.util.dir_contains_only(args.golden_dump_path, ".npy"):
            self.parse_tool.do_compare_converted_dir(args)
        else:
            self.parse_tool.do_vector_compare(args)

    def do_dc(self, line=""):
        self.parse_tool.do_convert_dump(self._parse_argv(line))

    def do_pt(self, line=""):
        self.parse_tool.do_print_data(self._parse_argv(line))

    def do_pk(self, line=""):
        self.parse_tool.do_parse_pkl(self._parse_argv(line))

    def do_cn(self, line=''):
        self.parse_tool.do_compare_data(self._parse_argv(line))

    def do_cad(self, line=''):
        self.parse_tool.do_convert_api_dir(self._parse_argv(line))
