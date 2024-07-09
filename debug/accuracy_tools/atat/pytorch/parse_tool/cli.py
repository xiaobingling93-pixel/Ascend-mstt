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
from .lib.interactive_cli import InteractiveCli


def _run_interactive_cli(cli=None):
    print("Interactive command mode")
    if not cli:
        cli = InteractiveCli()
    try:
        cli.cmdloop(intro="Start Parsing........")
    except KeyboardInterrupt:
        print("Exit parsing.......")


def parse():
    _run_interactive_cli()
