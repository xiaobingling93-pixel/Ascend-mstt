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
from atat.pytorch.parse_tool.lib.interactive_cli import InteractiveCli
from atat.core.utils import print_info_log


def _run_interactive_cli(cli=None):
    print_info_log("Interactive command mode")
    if not cli:
        cli = InteractiveCli()
    try:
        cli.cmdloop(intro="Start Parsing........")
    except KeyboardInterrupt:
        print_info_log("Exit parsing.......")


def parse():
    _run_interactive_cli()
