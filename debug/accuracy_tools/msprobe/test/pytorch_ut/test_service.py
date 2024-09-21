import unittest
from unittest.mock import patch, mock_open

import torch.nn as nn
from msprobe.core.common.utils import Const
from msprobe.pytorch.debugger.debugger_config import DebuggerConfig
from msprobe.pytorch.pt_config import parse_json_config
from msprobe.pytorch.service import Service


class TestService(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./dump/",
        }
        with patch("msprobe.pytorch.pt_config.FileOpen", mock_open(read_data='')), \
                patch("msprobe.pytorch.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        self.config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./ut_dump", "L1")
        self.service = Service(self.config)

    def test_start(self):
        with patch("msprobe.pytorch.service.get_rank_if_initialized", return_value=0), \
                patch("msprobe.pytorch.service.Service.create_dirs", return_value=None):
            self.service.start(None)
        self.assertEqual(self.service.current_rank, 0)

    def test_stop_and_step(self):
        with patch("msprobe.core.data_dump.data_collector.DataCollector.write_json", return_value=None):
            self.service.stop()
        self.assertFalse(self.service.switch)

        self.service.step()
        self.assertEqual(self.service.current_iter, 1)

    def test_register_hook_new(self):
        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(in_features=8, out_features=4)

            def forward(self, x):
                x = self.linear(x)
                return x

        self.service.model = TestModule()
        self.config.level = "L0"
        with patch("msprobe.pytorch.service.logger.info_on_rank_0") as mock_logger, \
                patch("msprobe.pytorch.service.remove_dropout", return_value=None):
            self.service.register_hook_new()
            self.assertEqual(mock_logger.call_count, 2)

    def test_create_dirs(self):
        with patch("msprobe.pytorch.service.create_directory"), \
                patch("msprobe.core.data_dump.data_collector.DataCollector.update_dump_paths"):
            self.service.create_dirs()
        self.assertEqual(self.service.dump_iter_dir, "./ut_dump/step0")

    def test_need_end_service(self):
        self.service.should_stop_service = True
        self.assertTrue(self.service.need_stop_service())

        self.service.should_stop_service = False
        self.service.config.step = [1, 3]
        self.service.current_iter = 1
        self.assertFalse(self.service.need_stop_service())

        self.service.current_iter = 2
        self.assertTrue(self.service.need_stop_service())

        self.service.current_iter = 4
        self.service.config.level = "L0"
        self.service.config.online_run_ut = False
        self.assertTrue(self.service.need_stop_service())
        self.assertFalse(self.service.switch)
        self.assertTrue(self.service.should_stop_service)

    def test_should_execute_hook(self):
        self.service.switch = True
        self.service.data_collector = None
        self.assertTrue(self.service.should_execute_hook())

        self.service.switch = False
        self.assertFalse(self.service.should_execute_hook())

        class DataProcessor:
            def __init__(self):
                self.is_terminated = True

        class DataCollector:
            def __init__(self):
                self.data_processor = DataProcessor()

        self.service.switch = True
        self.service.data_collector = DataCollector()
        self.assertFalse(self.service.should_execute_hook())
