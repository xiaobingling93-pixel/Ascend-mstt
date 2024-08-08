from unittest import TestCase
from unittest.mock import patch, mock_open

from msprobe.core.common.const import Const
from msprobe.pytorch.pt_config import parse_json_config, parse_task_config


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
        with patch("msprobe.pytorch.pt_config.os.path.join", return_value="/path/config.json"), \
                patch("msprobe.pytorch.pt_config.FileOpen", mock_open(read_data='')), \
                patch("msprobe.pytorch.pt_config.json.load", return_value=mock_json_data):
            common_config, task_config = parse_json_config(None, None)
        self.assertEqual(common_config.task, Const.STATISTICS)
        self.assertEqual(task_config.data_mode, ["all"])

        with patch("msprobe.pytorch.pt_config.os.path.join", return_value="/path/config.json"), \
                patch("msprobe.pytorch.pt_config.FileOpen", mock_open(read_data='')), \
                patch("msprobe.pytorch.pt_config.json.load", return_value=mock_json_data):
            common_config, task_config = parse_json_config(None, Const.TENSOR)
        self.assertEqual(common_config.task, Const.STATISTICS)
        self.assertEqual(task_config.file_format, "npy")

    def test_parse_task_config(self):
        overflow_check_config = {
            "overflow_check": {
                "overflow_nums": 1,
                "check_mode": "all"
            }
        }
        result = parse_task_config(Const.OVERFLOW_CHECK, overflow_check_config)
        self.assertEqual(result.overflow_nums, 1)
        self.assertEqual(result.check_mode, "all")

        free_benchmark_config = {
            "free_benchmark": {
                "scope": [],
                "list": ["conv2d"],
                "fuzz_device": "npu",
                "pert_mode": "improve_precision",
                "handler_type": "check",
                "fuzz_level": "L1",
                "fuzz_stage": "forward",
                "if_preheat": False,
                "preheat_step": 15,
                "max_sample": 20
            }
        }
        result = parse_task_config(Const.FREE_BENCHMARK, free_benchmark_config)
        self.assertEqual(result.pert_mode, "improve_precision")
        self.assertEqual(result.handler_type, "check")
        self.assertEqual(result.preheat_step, 15)
        self.assertEqual(result.max_sample, 20)
        
        run_ut_config = {
            "run_ut": {
                "white_list": ["conv2d"],
                "black_list": ["matmul"],
                "error_data_path": '/home/dump_path'
                
            }
        }
        with patch('os.path.exists', return_value=True) as mocked_exists:
            result = parse_task_config(Const.RUN_UT, run_ut_config)
            self.assertEqual(result.white_list, ["conv2d"])
            self.assertEqual(result.black_list, ["matmul"])
            self.assertEqual(result.error_data_path, '/home/dump_path')
            mocked_exists.assert_called_once_with('/home/dump_path')
