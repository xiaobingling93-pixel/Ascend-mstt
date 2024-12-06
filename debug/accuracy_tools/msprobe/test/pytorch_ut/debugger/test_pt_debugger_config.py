import unittest
from unittest.mock import MagicMock

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException
from msprobe.pytorch.debugger.debugger_config import DebuggerConfig


class TestDebuggerConfig(unittest.TestCase):
    def setUp(self):
        self.common_config = MagicMock()
        self.task_config = MagicMock()

        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.STATISTICS
        self.common_config.level = "L1"
        self.common_config.enable_dataloader = True

    def test_default_init(self):
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        self.assertEqual(debugger.dump_path, "./dump_path")
        self.assertEqual(debugger.task, Const.STATISTICS)
        self.assertEqual(debugger.level, "L1")
        self.assertTrue(debugger.enable_dataloader)

    def test_task_free_benchmark_init(self):
        self.task_config.fuzz_device = "npu"
        self.task_config.handler_type = "check"
        self.task_config.if_preheat = True

        debugger = DebuggerConfig(self.common_config, self.task_config, Const.FREE_BENCHMARK, None, None)
        self.assertEqual(debugger.fuzz_device, "npu")
        self.assertEqual(debugger.handler_type, "check")
        self.assertTrue(debugger.preheat_config["if_preheat"])

    def test_online_run_ut_initialization(self):
        self.task_config.online_run_ut = True
        self.task_config.nfs_path = "./nfs_path"
        self.task_config.tls_path = "./tls_path"
        self.task_config.host = "localhost"
        self.task_config.port = 8080

        debugger = DebuggerConfig(self.common_config, self.task_config, Const.TENSOR, None, None)
        self.assertTrue(debugger.online_run_ut)
        self.assertEqual(debugger.nfs_path, "./nfs_path")
        self.assertEqual(debugger.port, 8080)

    def test_valid_task_and_level(self):
        config = DebuggerConfig(self.common_config, self.task_config, "tensor", None, "L1")
        config.check_kwargs()

    def test_invalid_task(self):
        with self.assertRaises(MsprobeException) as context:
            config = DebuggerConfig(self.common_config, self.task_config, "invalid_task", None, "L1")
            config.check_kwargs()
        self.assertIn("not in the", str(context.exception))

    def test_invalid_level(self):
        with self.assertRaises(MsprobeException) as context:
            config = DebuggerConfig(self.common_config, self.task_config, "tensor", None, "invalid_level")
            config.check_kwargs()
        self.assertIn("not in the", str(context.exception))

    def test_missing_dump_path(self):
        with self.assertRaises(MsprobeException) as context:
            self.common_config.dump_path = None
            config = DebuggerConfig(self.common_config, self.task_config, "tensor", None, "L1")
            config.check_kwargs()
        self.assertIn("dump_path not found", str(context.exception))

    def test_check_and_adjust_config_with_l2_scope_not_empty(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.TENSOR

        self.task_config.scope = ["test_api_name"]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        with self.assertRaises(MsprobeException) as context:
            debugger._check_and_adjust_config_with_l2()
        self.assertIn("the scope cannot be configured", str(context.exception))

    def test_check_and_adjust_config_with_l2_list_empty(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.TENSOR

        self.task_config.scope = []
        self.task_config.list = []
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        with self.assertRaises(MsprobeException) as context:
            debugger._check_and_adjust_config_with_l2()
        self.assertIn("the list must be configured", str(context.exception))

    def test_check_and_adjust_config_with_l2_success(self):
        self.common_config.dump_path = "./dump_path"
        self.common_config.task = Const.TENSOR

        self.task_config.scope = []
        self.task_config.list = ["Functional.conv2d.0.backward"]
        debugger = DebuggerConfig(self.common_config, self.task_config, None, None, None)
        debugger._check_and_adjust_config_with_l2()
        self.assertIn("Functional.conv2d.0.forward", self.task_config.list)
