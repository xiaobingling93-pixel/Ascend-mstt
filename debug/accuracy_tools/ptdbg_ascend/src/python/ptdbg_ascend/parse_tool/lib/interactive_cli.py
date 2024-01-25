#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Copyright (C) 2022-2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import cmd
from .parse_tool import ParseTool
from .utils import Util
from .config import Const
from .parse_exception import catch_exception


class InteractiveCli(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = "Parse >>> "
        self.parse_tool = ParseTool()
        self.util = Util()
        self.util.print_panel(Const.HEADER)
        self._prepare()

    @staticmethod
    def _parse_argv(line, insert=None):
        argv = line.split() if line != "" else []
        if "-h" in argv:
            return argv
        if insert is not None and len(argv) and argv[0] != insert:
            argv.insert(0, insert)
        return argv

    def _prepare(self):
        self.parse_tool.prepare()

    @catch_exception
    def default(self, line=""):
        self.util.execute_command(line)
        return False

    @catch_exception
    def do_run(self, line=""):
        self.util.execute_command(line)

    def do_vc(self, line=""):
        self.parse_tool.do_vector_compare(self._parse_argv(line))

    def do_dc(self, line=""):
        self.parse_tool.do_convert_dump(self._parse_argv(line))

    def do_pt(self, line=""):
        self.parse_tool.do_print_data(self._parse_argv(line))

    def do_pk(self, line=""):
        self.parse_tool.do_parse_pkl(self._parse_argv(line))

    def do_cn(self, line=''):
        self.parse_tool.do_compare_data(self._parse_argv(line))
