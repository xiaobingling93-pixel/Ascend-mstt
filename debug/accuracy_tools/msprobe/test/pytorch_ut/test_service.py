# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import unittest
from unittest.mock import patch, mock_open, MagicMock
from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import Const
from msprobe.core.data_dump.api_registry import ApiRegistry
from msprobe.pytorch.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.pt_config import parse_json_config
from msprobe.pytorch.pytorch_service import PytorchService


class TestService(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./dump/",
        }
        with patch("msprobe.pytorch.pt_config.FileOpen", mock_open(read_data='')), \
                patch("msprobe.pytorch.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        self.config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./ut_dump", "L1")
        self.service = PytorchService(self.config)

    def test_start_success(self):
        with patch("msprobe.pytorch.service.get_rank_if_initialized", return_value=0), \
                patch("msprobe.pytorch.service.Service.create_dirs", return_value=None):
            self.service.start(None)
        self.assertEqual(self.service.current_rank, 0)

    def test_start_fail(self):
        self.service.config.rank = [1, 2]
        self.service.current_rank = 3
        self.assertIsNone(self.service.start(None))

        self.service.config.step = [1, 2]
        self.service.current_iter = 3
        self.assertIsNone(self.service.start(None))

    @patch("msprobe.core.data_dump.data_collector.DataCollector.write_json")
    def test_stop_success(self, mock_write_json):
        mock_write_json.return_value = None
        self.service.stop()

        self.assertFalse(Runtime.is_running)

    def test_stop_fail(self):
        Runtime.is_running = True

        self.service.config.rank = [1, 2]
        self.service.current_rank = 3
        res = self.service.stop()
        self.assertIsNone(res)
        self.assertTrue(Runtime.is_running)

        self.service.config.step = [1, 2]
        self.service.current_iter = 3
        res = self.service.stop()
        self.assertIsNone(res)
        self.assertTrue(Runtime.is_running)

        self.service.config.level = "L2"
        res = self.service.stop()
        self.assertIsNone(res)
        self.assertTrue(Runtime.is_running)

        self.service.should_stop_service = True
        res = self.service.stop()
        self.assertIsNone(res)
        self.assertTrue(Runtime.is_running)

    def test_step_success(self):
        self.service.step()
        self.assertEqual(self.service.loop, 1)

    def test_step_fail(self):
        self.service.should_stop_service = True
        self.assertIsNone(self.service.step())

    def test_register_module_hook_with_level0(self):
        self.service.model = MagicMock()
        self.service.build_hook = MagicMock()
        self.config.level = "L0"
        with patch("msprobe.pytorch.service.logger.info_on_rank_0") as mock_logger, \
                patch("msprobe.pytorch.service.ModuleProcesser.register_module_hook") as mock_register_module_hook:
            self.service._register_module_hook()
            self.assertEqual(mock_logger.call_count, 1)
            mock_register_module_hook.assert_called_once()

    def test_register_api_hook_with_level1(self):
        self.service.build_hook = MagicMock()
        self.config.level = "L1"
        with patch("msprobe.pytorch.service.logger.info_on_rank_0") as mock_logger, \
             patch.object(ApiRegistry, "initialize_hook") as mock_init_hook, \
             patch.object(ApiRegistry, 'register_all_api') as mock_api_modularity:
            self.service._register_api_hook()
            self.assertEqual(mock_logger.call_count, 1)
            mock_init_hook.assert_called_once()
            mock_api_modularity.assert_called_once()

    def test_create_dirs(self):
        with patch("msprobe.pytorch.service.create_directory"), \
                patch("msprobe.core.data_dump.data_collector.DataCollector.update_dump_paths"), \
                patch("msprobe.core.data_dump.data_collector.DataCollector.initialize_json_file"):
            self.service.create_dirs()
        self.assertEqual(self.service.dump_iter_dir, "./ut_dump/step0")

    def test_need_end_service(self):
        self.service.should_stop_service = True
        self.assertTrue(self.service._need_stop_service())

        self.service.should_stop_service = False
        self.service.config.step = [1, 3]
        self.service.current_iter = 1
        self.assertFalse(self.service._need_stop_service())

        self.service.current_iter = 2
        self.assertTrue(self.service._need_stop_service())

        self.service.current_iter = 4
        self.service.config.level = "L0"
        self.service.config.online_run_ut = False
        self.assertTrue(self.service._need_stop_service())
        self.assertFalse(Runtime.is_running)
        self.assertTrue(self.service.should_stop_service)
