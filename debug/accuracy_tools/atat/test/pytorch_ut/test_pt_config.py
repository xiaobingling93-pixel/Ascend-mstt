from unittest import TestCase
from unittest.mock import patch, mock_open

from atat.core.common.utils import Const
from atat.pytorch.pt_config import parse_json_config


class TestPtConfig(TestCase):
    def test_parse_json_config(self):
        mock_json_data = {
            "task": "statistics",
            "dump_path": "./dump/",
            "rank": [],
            "step": [],
            "level": "L1",
            "seed": 1234,
            "statistics": {
                "scope": [],
                "list": [],
                "data_mode": ["all"],
            },
            "tensor": {
                "file_format": "npy"
            }
        }
        with patch("atat.pytorch.pt_config.os.path.join", return_value="/path/config.json"), \
                patch("atat.pytorch.pt_config.FileOpen", mock_open(read_data='')), \
                patch("atat.pytorch.pt_config.json.load", return_value=mock_json_data):
            common_config, task_config = parse_json_config(None, None)
        self.assertEqual(common_config.task, Const.STATISTICS)
        self.assertEqual(task_config.data_mode, ["all"])

        with patch("atat.pytorch.pt_config.os.path.join", return_value="/path/config.json"), \
                patch("atat.pytorch.pt_config.FileOpen", mock_open(read_data='')), \
                patch("atat.pytorch.pt_config.json.load", return_value=mock_json_data):
            common_config, task_config = parse_json_config(None, Const.TENSOR)
        self.assertEqual(common_config.task, Const.STATISTICS)
        self.assertEqual(task_config.file_format, "npy")
