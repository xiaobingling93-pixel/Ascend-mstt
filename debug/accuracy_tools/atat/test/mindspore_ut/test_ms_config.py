from unittest import TestCase
from unittest.mock import patch, mock_open

from atat.core.common.const import Const
from atat.mindspore.ms_config import parse_json_config


class TestMsConfig(TestCase):
    def test_parse_json_config(self):
        mock_json_data = {
            "dump_path": "./dump/",
            "rank": [],
            "step": [],
            "level": "L1",
            "seed": 1234,
            "statistics": {
                "scope": [],
                "list": [],
                "data_mode": ["all"],
                "summary_mode": "statistics"
            }
        }
        with patch("atat.mindspore.ms_config.FileOpen", mock_open(read_data='')), \
                patch("atat.mindspore.ms_config.json.load", return_value=mock_json_data):
            common_config, task_config = parse_json_config("./config.json")
        self.assertEqual(common_config.task, Const.STATISTICS)
        self.assertEqual(task_config.data_mode, ["all"])

        with self.assertRaises(Exception) as context:
            parse_json_config(None)
        self.assertEqual(str(context.exception), "json file path is None")
