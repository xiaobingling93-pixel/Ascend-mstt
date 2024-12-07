# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

from msprobe.pytorch.parse_tool.lib.interactive_cli import InteractiveCli
from msprobe.pytorch.common.log import logger


def _run_interactive_cli(cli=None):
    logger.info("Interactive command mode")
    if not cli:
        cli = InteractiveCli()
    try:
        cli.cmdloop(intro="Start Parsing........")
    except KeyboardInterrupt:
        logger.info("Exit parsing.......")


def parse():
    _run_interactive_cli()
