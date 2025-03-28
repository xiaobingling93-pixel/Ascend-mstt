# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
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

import os
import unittest
from unittest.mock import patch

from msprobe.core.common.log import logger
from msprobe.mindspore.free_benchmark.self_check_tool_factory import SelfCheckToolFactory
from msprobe.mindspore.free_benchmark.api_pynative_self_check import ApiPyNativeSelfCheck
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.common.const import Const as MsConst
from msprobe.core.common.const import Const


class TestSelfCheckToolFactory(unittest.TestCase):

    @patch.object(logger, "error")
    def test_create(self, mock_logger_error):
        common_config = CommonConfig({})
        common_config.task = Const.FREE_BENCHMARK
        common_config.dump_path = os.path.dirname(os.path.realpath(__file__))
        task_config = BaseConfig({})
        config = DebuggerConfig(common_config, task_config)

        config.level = "UNKNOWN"
        with self.assertRaises(ValueError):
            SelfCheckToolFactory.create(config)
        mock_logger_error.assert_called_with("UNKNOWN is not supported.")
        mock_logger_error.reset_mock()

        config.level = MsConst.API
        config.execution_mode = MsConst.GRAPH_KBYK_MODE
        with self.assertRaises(ValueError):
            SelfCheckToolFactory.create(config)
        mock_logger_error.assert_called_with(f"Task free_benchmark is not supported in this mode: {MsConst.GRAPH_KBYK_MODE}.")

        config.execution_mode = MsConst.PYNATIVE_MODE
        tool = SelfCheckToolFactory.create(config)
        self.assertIsInstance(tool, ApiPyNativeSelfCheck, "")
