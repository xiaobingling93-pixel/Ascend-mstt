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


import unittest
from unittest.mock import patch

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common_config import CommonConfig
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.ms_config import StatisticsConfig


class TestDebuggerConfig(unittest.TestCase):
    @patch.object(logger, "error")
    @patch("msprobe.mindspore.debugger.debugger_config.create_directory")
    def test_init(self, _, mock_logger_error):
        json_config = {
            "dump_path": "/absolute_path",
            "rank": [],
            "step": [],
            "level": "L2"
        }
        common_config = CommonConfig(json_config)
        task_config = StatisticsConfig(json_config)
        debugger_config = DebuggerConfig(common_config, task_config)
        self.assertEqual(debugger_config.task, Const.STATISTICS)
        self.assertEqual(debugger_config.file_format, "npy")
        self.assertEqual(debugger_config.check_mode, "all")
        self.assertEqual(debugger_config.overflow_nums, 1)
        self.assertEqual(debugger_config.tensor_list, [])

        common_config.level = "L1"
        common_config.task = Const.FREE_BENCHMARK
        debugger_config = DebuggerConfig(common_config, task_config)
        self.assertEqual(debugger_config.pert_type, FreeBenchmarkConst.DEFAULT_PERT_TYPE)
        self.assertEqual(debugger_config.handler_type, FreeBenchmarkConst.DEFAULT_HANDLER_TYPE)
        self.assertEqual(debugger_config.dump_level, FreeBenchmarkConst.DEFAULT_DUMP_LEVEL)
        self.assertEqual(debugger_config.stage, FreeBenchmarkConst.DEFAULT_STAGE)

        task_config.handler_type = FreeBenchmarkConst.FIX
        task_config.pert_mode = FreeBenchmarkConst.ADD_NOISE
        with self.assertRaises(ValueError):
            DebuggerConfig(common_config, task_config)
        mock_logger_error.assert_called_with("pert_mode must be improve_precision or empty when handler_type is fix, "
                                             f"but got {FreeBenchmarkConst.ADD_NOISE}.")
        mock_logger_error.reset_mock()

        task_config.handler_type = FreeBenchmarkConst.FIX
        task_config.pert_mode = FreeBenchmarkConst.DEFAULT_PERT_TYPE
        task_config.fuzz_stage = Const.BACKWARD
        with self.assertRaises(ValueError):
            DebuggerConfig(common_config, task_config)
        mock_logger_error.assert_called_with("handler_type must be check or empty when fuzz_stage is backward, "
                                             f"but got {task_config.handler_type}.")
