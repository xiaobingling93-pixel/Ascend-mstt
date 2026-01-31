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

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.handler.check_handler import CheckHandler
from msprobe.mindspore.free_benchmark.handler.fix_handler import FixHandler
from msprobe.mindspore.free_benchmark.handler.handler_factory import HandlerFactory


class TestHandlerFactory(unittest.TestCase):
    @patch.object(logger, "error")
    def test_create(self, mock_error):
        api_name_with_id = "Mint.add.0"

        Config.handler_type = "UNKNOWN"
        with self.assertRaises(Exception):
            HandlerFactory.create(api_name_with_id)
            mock_error.assert_called_with("UNKNOWN is not supported.")

        Config.handler_type = FreeBenchmarkConst.CHECK
        handler = HandlerFactory.create(api_name_with_id)
        self.assertTrue(isinstance(handler, CheckHandler))
        self.assertEqual(handler.api_name_with_id, api_name_with_id)

        Config.handler_type = FreeBenchmarkConst.FIX
        handler = HandlerFactory.create(api_name_with_id)
        self.assertTrue(isinstance(handler, FixHandler))
        self.assertEqual(handler.api_name_with_id, api_name_with_id)
