import os
import shutil
import unittest
from unittest.mock import patch

from msprobe.core.common.const import Const
from msprobe.pytorch.pt_config import parse_json_config, parse_task_config, TensorConfig, \
    StatisticsConfig, OverflowCheckConfig, FreeBenchmarkCheckConfig, RunUTConfig, GradToolConfig


class TestPtConfig(unittest.TestCase):
    def test_parse_json_config(self):
        mock_json_data = {
            "task": "statistics",
            "dump_path": "./dump/",
            "rank": [],
            "step": [],
            "level": "L1",
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
                patch("msprobe.pytorch.pt_config.load_json", return_value=mock_json_data):
            common_config, task_config = parse_json_config(None, None)
        self.assertEqual(common_config.task, Const.STATISTICS)
        self.assertEqual(task_config.data_mode, [Const.ALL])

        with patch("msprobe.pytorch.pt_config.os.path.join", return_value="/path/config.json"), \
                patch("msprobe.pytorch.pt_config.load_json", return_value=mock_json_data):
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
        self.assertEqual(result.check_mode, Const.ALL)

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


class TestTensorConfig(unittest.TestCase):

    def setUp(self):
        self.json_config = {
            "online_run_ut": False,
            "host": "127.0.0.1",
            "port": 8080
        }
        self.config = TensorConfig(self.json_config)

    def test_check_file_format_valid(self):
        self.config.file_format = "npy"
        self.config._check_file_format()

        self.config.file_format = "bin"
        self.config._check_file_format()

    def test_check_file_format_invalid(self):
        self.config.file_format = "invalid_format"
        with self.assertRaises(Exception) as context:
            self.config._check_file_format()
        self.assertIn(str(context.exception), "file_format is invalid")

    @patch('msprobe.pytorch.pt_config.check_crt_valid')
    def test_check_online_run_ut(self, mock_check_crt_valid):
        mock_check_crt_valid.return_value = True

        self.config.online_run_ut = "True"
        with self.assertRaises(Exception) as context:
            self.config._check_online_run_ut()
        self.assertIn(str(context.exception), f"online_run_ut: {self.config.online_run_ut} is invalid.")
        self.config.online_run_ut = True

        self.config.online_run_ut_recompute = "True"
        with self.assertRaises(Exception) as context:
            self.config._check_online_run_ut()
        self.assertIn(str(context.exception), f"online_run_ut_recompute: {self.config.online_run_ut} is invalid.")
        self.config.online_run_ut_recompute = False

        self.config.nfs_path = "./nfs_path"
        with self.assertRaises(Exception) as context:
            self.config._check_online_run_ut()
        self.assertIn(str(context.exception), "[msprobe] 非法文件路径： ")
        self.config.nfs_path = ""

        self.config.tls_path = "./tls_path"
        with self.assertRaises(Exception) as context:
            self.config._check_online_run_ut()
        self.assertIn(str(context.exception), "[msprobe] 非法文件路径： ")

        os.makedirs(self.config.tls_path)
        with open(os.path.join(self.config.tls_path, "client.key"), 'w') as file:
            file.write("1")
        with open(os.path.join(self.config.tls_path, "client.crt"), 'w') as file:
            file.write("1")
        self.config._check_online_run_ut()
        shutil.rmtree(self.config.tls_path)
        self.config.tls_path = ""

        self.config.host = "invalid_host"
        with self.assertRaises(Exception) as context:
            self.config._check_online_run_ut()
        self.assertIn(str(context.exception), f"host: {self.config.host} is invalid.")
        self.config.host = "127.0.0.1"

        self.config.port = -1
        with self.assertRaises(Exception) as context:
            self.config._check_online_run_ut()
        self.assertIn(str(context.exception), f"port: {self.config.port} is invalid, port range 0-65535.")
        self.config.port = 6123

        # all config right
        self.config._check_online_run_ut()


class TestStatisticsConfig(unittest.TestCase):

    def setUp(self):
        self.json_config = {}
        self.config = StatisticsConfig(self.json_config)

    def test_check_summary_mode_valid_statistics(self):
        self.config.summary_mode = Const.STATISTICS
        try:
            self.config._check_summary_mode()
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_check_summary_mode_valid_md5(self):
        self.config.summary_mode = Const.MD5
        try:
            self.config._check_summary_mode()
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_check_summary_mode_invalid(self):
        self.config.summary_mode = "invalid_mode"
        with self.assertRaises(Exception) as context:
            self.config._check_summary_mode()
        self.assertIn(str(context.exception), "[msprobe] 无效参数：")

    def test_check_summary_mode_none(self):
        self.config.summary_mode = None
        try:
            self.config._check_summary_mode()
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")


class TestOverflowCheckConfig(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            "overflow_nums": 2,
            "check_mode": "all"
        }
        self.invalid_overflow_nums_config_str = {
            "overflow_nums": "not_an_int",
            "check_mode": "all"
        }
        self.invalid_overflow_nums_config_bool = {
            "overflow_nums": bool,
            "check_mode": "all"
        }
        self.invalid_check_mode_config = {
            "overflow_nums": 2,
            "check_mode": "invalid_mode"
        }

    def test_valid_config(self):
        config = OverflowCheckConfig(self.valid_config)
        self.assertEqual(config.overflow_nums, 2)
        self.assertEqual(config.check_mode, Const.ALL)

    def test_invalid_overflow_nums_str_type(self):
        with self.assertRaises(Exception) as context:
            OverflowCheckConfig(self.invalid_overflow_nums_config_str)
        self.assertEqual(str(context.exception), "overflow_num is invalid")

    def test_invalid_overflow_nums_bool_type(self):
        with self.assertRaises(Exception) as context:
            OverflowCheckConfig(self.invalid_overflow_nums_config_bool)
        self.assertEqual(str(context.exception), "overflow_num is invalid")

    def test_invalid_check_mode(self):
        with self.assertRaises(Exception) as context:
            OverflowCheckConfig(self.invalid_check_mode_config)
        self.assertEqual(str(context.exception), "check_mode is invalid")


class TestFreeBenchmarkCheckConfig(unittest.TestCase):

    def setUp(self):
        self.valid_config = {
            "fuzz_device": "npu",
            "pert_mode": "improve_precision",
            "handler_type": "check",
            "fuzz_level": "L1",
            "fuzz_stage": "forward",
            "if_preheat": True,
            "preheat_step": 15,
            "max_sample": 20
        }
        self.config = FreeBenchmarkCheckConfig(self.valid_config)

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_pert_mode_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["pert_mode"] = "INVALID_MODE"
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("pert_mode is invalid", str(mock_error.call_args))

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_fuzz_device_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["fuzz_device"] = "INVALID_DEVICE"
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("fuzz_device is invalid", str(mock_error.call_args))
        
    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_fuzz_device_cpu_mode_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["fuzz_device"] = "cpu"
        invalid_config["pert_mode"] = "INVALID_CPU_MODE"
        config = FreeBenchmarkCheckConfig(invalid_config)
        self.assertIn("You need to and can only set fuzz_device as ", str(mock_error.call_args))

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_handler_type_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["handler_type"] = "INVALID_HANDLER"
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("handler_type is invalid", str(mock_error.call_args))
        
    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_fuzz_stage_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["fuzz_stage"] = "INVALID_STAGE"
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("fuzz_stage is invalid", str(mock_error.call_args))

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_fuzz_level_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["fuzz_level"] = "INVALID_LEVEL"
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("fuzz_level is invalid", str(mock_error.call_args))

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_if_preheat_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["if_preheat"] = "not_a_bool"
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("if_preheat is invalid", str(mock_error.call_args))

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_preheat_step_invalid_not_int(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["if_preheat"] = True
        invalid_config["preheat_step"] = 3.5
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("preheat_step is invalid, it should be an integer", str(mock_error.call_args))

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_preheat_step_invalid_not_great_than_zero(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["if_preheat"] = True
        invalid_config["preheat_step"] = -5
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("preheat_step must be greater than 0", str(mock_error.call_args))
        
    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_preheat_max_sample_not_int(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["if_preheat"] = True
        invalid_config["max_sample"] = 3.5
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("max_sample is invalid, it should be an integer", str(mock_error.call_args))
        
    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_max_sample_invalid_not_great_than_zero(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["if_preheat"] = True
        invalid_config["max_sample"] = -5
        config = FreeBenchmarkCheckConfig(invalid_config)
        mock_error.assert_called_once()
        self.assertIn("max_sample must be greater than 0", str(mock_error.call_args))

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_fix_config_preheat_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["if_preheat"] = True
        config = FreeBenchmarkCheckConfig(invalid_config)
        config._check_fix_config()
        self.assertIn("Preheating is not supported for fix handler type", str(mock_error.call_args))

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_fix_stage_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["fuzz_stage"] = "INVALID_STAGE"
        config = FreeBenchmarkCheckConfig(invalid_config)
        config._check_fix_config()
        self.assertIn("The fuzz_stage when opening fix handler must be one of", str(mock_error.call_args))

    @patch('msprobe.core.common.log.logger.error_log_with_exp')
    def test_check_fix_mode_invalid(self, mock_error):
        invalid_config = self.valid_config.copy()
        invalid_config["pert_mode"] = "INVALID_MODE"
        config = FreeBenchmarkCheckConfig(invalid_config)
        config._check_fix_config()
        self.assertIn("The pert_mode when opening fix handler must be one of", str(mock_error.call_args))


class TestRunUTConfig(unittest.TestCase):

    @patch('msprobe.pytorch.hook_module.utils.get_ops', return_value=['relu', 'gelu', 'conv2d'])
    def setUp(self, mock_get_ops):
        self.config = RunUTConfig({
            "white_list": ["relu"],
            "black_list": ["gelu"]
        })

    def test_check_filter_list_config_invalid_type(self):
        with self.assertRaises(Exception) as context:
            RunUTConfig.check_filter_list_config(Const.WHITE_LIST, "not_a_list")
        self.assertIn("must be a list type", str(context.exception))

    def test_check_filter_list_element_config_invalid_type(self):
        with self.assertRaises(Exception) as context:
            RunUTConfig.check_filter_list_config("white_list", [1, 1])
        self.assertIn("All elements in ", str(context.exception))

    def test_check_filter_list_config_invalid_item(self):
        with self.assertRaises(Exception) as context:
            RunUTConfig.check_filter_list_config("white_list", ["api1"])
        self.assertIn("Invalid api in white_list:", str(context.exception))

    @patch('os.path.exists', return_value=False)
    def test_check_error_data_path_config_not_exist(self, mock_exists):
        with self.assertRaises(Exception) as context:
            RunUTConfig.check_error_data_path_config("./invalid_path")
        self.assertIn("does not exist", str(context.exception))

    @patch('os.path.exists', return_value=False)
    def test_check_nfs_path_config_not_exist(self, mock_exists):
        with self.assertRaises(Exception) as context:
            RunUTConfig.check_nfs_path_config("./invalid_nfs")
        self.assertIn("[msprobe] 非法文件路径：", str(context.exception))

    @patch('os.path.exists', return_value=False)
    def test_check_tls_path_config_not_exist(self, mock_exists):
        with self.assertRaises(Exception) as context:
            RunUTConfig.check_tls_path_config("./invalid_tls")
        self.assertIn("[msprobe] 非法文件路径：", str(context.exception))

    def test_check_run_ut_config(self):
        with patch.object(RunUTConfig, 'check_filter_list_config') as mock_filter, \
             patch.object(RunUTConfig, 'check_error_data_path_config') as mock_error, \
             patch.object(RunUTConfig, 'check_nfs_path_config') as mock_nfs, \
             patch.object(RunUTConfig, 'check_tls_path_config') as mock_tls:
            self.config.check_run_ut_config()
            mock_filter.assert_called()
            mock_error.assert_called()
            mock_nfs.assert_called()
            mock_tls.assert_called()


class TestGradToolConfig(unittest.TestCase):
    def setUp(self):
        self.level_adp = {"L1": None, "L2": None}
        global level_adp
        level_adp = self.level_adp

    def test_invalid_grad_level(self):
        json_config = {
            "grad_level": "invalid_level",
            "param_list": ["param1"],
            "bounds": [-1, 0, 1]
        }
        with self.assertRaises(Exception) as context:
            GradToolConfig(json_config)
        self.assertTrue("grad_level must be one of" in str(context.exception))

    def test_invalid_param_list(self):
        json_config = {
            "grad_level": "L1",
            "param_list": 1,
            "bounds": [-1, 0, 1]
        }
        with self.assertRaises(Exception) as context:
            GradToolConfig(json_config)
        self.assertTrue("param_list must be a list" in str(context.exception))
