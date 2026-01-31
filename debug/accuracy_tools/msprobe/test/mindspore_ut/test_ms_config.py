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
from msprobe.mindspore.ms_config import (parse_task_config,
                                         TensorConfig, StatisticsConfig, OverflowCheckConfig, FreeBenchmarkConfig)


class TestMsConfig(unittest.TestCase):

    def test_parse_task_config(self):
        mock_json_config = {
            "tensor": None,
            "statistics": None,
            "overflow_check": None,
            "free_benchmark": None
        }

        task_config = parse_task_config("tensor", mock_json_config)
        self.assertTrue(isinstance(task_config, TensorConfig))

        task_config = parse_task_config("statistics", mock_json_config)
        self.assertTrue(isinstance(task_config, StatisticsConfig))

        task_config = parse_task_config("overflow_check", mock_json_config)
        self.assertTrue(isinstance(task_config, OverflowCheckConfig))

        mock_json_config.update({"overflow_check": {"overflow_nums": "1"}})
        with self.assertRaises(Exception) as context:
            task_config = parse_task_config("overflow_check", mock_json_config)
        self.assertEqual(str(context.exception), "overflow_nums is invalid, it should be an integer")

        mock_json_config.update({"overflow_check": {"overflow_nums": 0}})
        with self.assertRaises(Exception) as context:
            task_config = parse_task_config("overflow_check", mock_json_config)
        self.assertEqual(str(context.exception), "overflow_nums should be -1 or positive integer")

        mock_json_config.update({"overflow_check": {"overflow_nums": 1}})
        mock_json_config.update({"overflow_check": {"check_mode": "core"}})
        with self.assertRaises(Exception) as context:
            task_config = parse_task_config("overflow_check", mock_json_config)
        self.assertEqual(str(context.exception), "check_mode is invalid")

        mock_json_config.update({"free_benchmark": {"fuzz_stage": Const.FORWARD}})
        task_config = parse_task_config("free_benchmark", mock_json_config)
        self.assertTrue(isinstance(task_config, FreeBenchmarkConfig))

        mock_json_config.update({"free_benchmark": {"fuzz_stage": Const.BACKWARD}})
        task_config = parse_task_config("free_benchmark", mock_json_config)
        self.assertTrue(isinstance(task_config, FreeBenchmarkConfig))

        mock_json_config.update({"free_benchmark": {"fuzz_stage": "unsupported_stage"}})
        with self.assertRaises(Exception) as context:
            task_config = parse_task_config("free_benchmark", mock_json_config)
        self.assertEqual(str(context.exception), "fuzz_stage must be forward, backward or empty")

        with self.assertRaises(Exception) as context:
            parse_task_config("unsupported_task", mock_json_config)
        self.assertEqual(str(context.exception), "task is invalid.")
