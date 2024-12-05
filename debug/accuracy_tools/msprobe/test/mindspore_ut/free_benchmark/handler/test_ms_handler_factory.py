# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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
