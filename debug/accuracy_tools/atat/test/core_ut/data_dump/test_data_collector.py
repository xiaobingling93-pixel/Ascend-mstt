import unittest
from unittest.mock import patch, mock_open, MagicMock

from atat.core.common.utils import Const
from atat.core.data_dump.data_collector import DataCollector
from atat.pytorch.debugger.debugger_config import DebuggerConfig
from atat.pytorch.pt_config import parse_json_config


class TestDataCollector(unittest.TestCase):
    def setUp(self):
        mock_json_data = {
            "dump_path": "./ut_dump",
        }
        with patch("atat.pytorch.pt_config.FileOpen", mock_open(read_data='')), \
                patch("atat.pytorch.pt_config.json.load", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json", Const.STATISTICS)
        config = DebuggerConfig(common_config, task_config, Const.STATISTICS, "./ut_dump", "L1")
        self.data_collector = DataCollector(config)

    def test_update_data(self):
        self.data_collector.config.task = Const.OVERFLOW_CHECK
        self.data_collector.data_processor.has_overflow = True
        with patch("atat.core.data_dump.json_writer.DataWriter.update_data", return_value=None):
            result1 = self.data_collector.update_data("test message", "test1:")
        self.assertEqual(result1, "test1:Overflow detected.")

        self.data_collector.data_processor.has_overflow = False
        result2 = self.data_collector.update_data("test message", "test2:")
        self.assertEqual(result2, "test2:No Overflow, OK.")

        self.data_collector.config.task = Const.STATISTICS
        self.data_collector.data_processor.has_overflow = True
        with patch("atat.core.data_dump.json_writer.DataWriter.update_data", return_value=None):
            result3 = self.data_collector.update_data("test message", "test3")
        self.assertEqual(result3, "test3")

    def test_pre_forward_data_collect(self):
        self.data_collector.check_scope_and_pid = MagicMock(return_value=False)
        self.data_collector.is_inplace = MagicMock(return_value=False)
        self.data_collector.data_processor.analyze_pre_forward = MagicMock()
        name = "TestModule.forward"
        pid = 123

        self.data_collector.pre_forward_data_collect(name, None, pid, None)
        self.data_collector.check_scope_and_pid.assert_called_once_with(
            self.data_collector.scope, "TestModule.backward", 123)
