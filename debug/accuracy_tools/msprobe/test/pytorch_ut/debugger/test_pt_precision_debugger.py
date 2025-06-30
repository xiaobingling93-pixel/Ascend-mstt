import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import torch

from msprobe.core.common.const import Const, MsgConst
from msprobe.core.common.utils import get_real_step_or_rank
from msprobe.core.common.exceptions import MsprobeException, FileCheckException
from msprobe.pytorch.debugger.precision_debugger import PrecisionDebugger
from msprobe.pytorch.grad_probe.grad_monitor import GradientMonitor
from msprobe.test.pytorch_ut.grad_probe.test_grad_monitor import common_config, task_config
from msprobe.core.common_config import CommonConfig
from msprobe.core.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.pytorch.pt_config import StatisticsConfig, GradToolConfig


class Args:
    def __init__(self, config_path=None, task=None, dump_path=None, level=None, model=None):
        self.config_path = config_path
        self.task = task
        self.dump_path = dump_path
        self.level = level
        self.model = model


class TestPrecisionDebugger(unittest.TestCase):
    grad_json_config = {
        "task": Const.GRAD_PROBE,
        "dump_path": "/absolute_path",
        "rank": [],
        "step": [],
        "level": "L1",
        "async_dump": False
    }

    grad_common_config = CommonConfig(grad_json_config)
    grad_task_config = GradToolConfig(grad_json_config)

    json_config = {
        "task": "statistics",
        "dump_path": "/absolute_path",
        "rank": [],
        "step": [],
        "level": "L1",
        "async_dump": False
    }

    statistics_common_config = CommonConfig(json_config)
    statistics_task_config = StatisticsConfig(json_config)

    def test_init(self):
        gm = GradientMonitor(common_config, task_config)
        self.assertIsNotNone(gm)
        step = get_real_step_or_rank([0, 1, "3-5"], Const.STEP)
        self.assertListEqual(step, [0, 1, 3, 4, 5])

    def test_check_input_params(self):
        args = Args(config_path=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(config_path="./")
        with self.assertRaises(FileCheckException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, FileCheckException.INVALID_FILE_ERROR)

        args = Args(task=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(dump_path=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(level=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(config_path=os.path.join(os.path.dirname(__file__), "../../../config.json"),
                    task=Const.TASK_LIST[0],
                    dump_path="./dump_path",
                    level=Const.LEVEL_LIST[0],
                    model=torch.nn.Module())
        checked_input_params = PrecisionDebugger._check_input_params(
            args.config_path,
            args.task,
            args.dump_path,
            args.level
        )
        self.assertIsNone(checked_input_params)

    def test_start_grad_probe(self):
        with self.assertRaises(Exception) as context:
            PrecisionDebugger._instance = None
            PrecisionDebugger.start()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)

        with patch.object(BasePrecisionDebugger, "_parse_config_path",
                          return_value=(self.grad_common_config, self.grad_task_config)):
            PrecisionDebugger._instance = PrecisionDebugger(task=Const.GRAD_PROBE, dump_path="./dump_path")
        checked_start = PrecisionDebugger.start()
        self.assertIsNone(checked_start)

    def test_start_statistics(self):
        PrecisionDebugger._instance = None
        with patch.object(BasePrecisionDebugger, "_parse_config_path",
                          return_value=(self.statistics_common_config, self.statistics_task_config)):
            debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.config = MagicMock()
        debugger.task = 'statistics'
        debugger.start()
        debugger.service.start.assert_called_once()

    def test_forward_backward_dump_end(self):
        with patch.object(
            BasePrecisionDebugger,
            "_parse_config_path",
            return_value=(self.statistics_common_config,self.statistics_task_config)
        ):
            debugger = PrecisionDebugger(dump_path="./dump_path", task='statistics')
        debugger.service = MagicMock()
        debugger.config = MagicMock()
        debugger.task = 'statistics'
        debugger.forward_backward_dump_end()
        debugger.service.stop.assert_called_once()

    def test_stop_grad_probe(self):
        with self.assertRaises(Exception) as context:
            PrecisionDebugger._instance = None
            PrecisionDebugger.stop()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)

        with patch.object(BasePrecisionDebugger, "_parse_config_path",
                          return_value=(self.grad_common_config, self.grad_task_config)):
            PrecisionDebugger._instance = PrecisionDebugger(task=Const.GRAD_PROBE, dump_path="./dump_path")
        checked_stop = PrecisionDebugger.stop()
        self.assertIsNone(checked_stop)

    def test_stop_statistics(self):
        PrecisionDebugger._instance = None
        debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.task = ''
        debugger.stop()
        debugger.service.stop.assert_called_once()

    def test_step_grad_probe(self):
        with self.assertRaises(Exception) as context:
            PrecisionDebugger._instance = None
            PrecisionDebugger.step()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)
        with patch.object(BasePrecisionDebugger, "_parse_config_path",
                          return_value=(self.grad_common_config, self.grad_task_config)):
            PrecisionDebugger._instance = PrecisionDebugger(task=Const.GRAD_PROBE, dump_path="./dump_path")
        checked_step = PrecisionDebugger.step()
        self.assertIsNone(checked_step)

    def test_step_statistics(self):
        debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.task = ''
        debugger.step()
        debugger.service.step.assert_called_once()

    def test_monitor(self):
        with self.assertRaises(Exception) as context:
            PrecisionDebugger._instance = None
            PrecisionDebugger.monitor(torch.nn.Module())
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)

        with patch.object(
            BasePrecisionDebugger,
            "_parse_config_path",
            return_value=(self.statistics_common_config, self.statistics_task_config)
        ):
            debugger = PrecisionDebugger(task=Const.STATISTICS, dump_path="./dump_path")
        checked_monitor = debugger.monitor(torch.nn.Module())
        self.assertIsNone(checked_monitor)

        debugger = PrecisionDebugger(task=Const.GRAD_PROBE, dump_path="./dump_path")
        debugger.gm = MagicMock()
        debugger.service = MagicMock()
        debugger.task = Const.GRAD_PROBE
        debugger.gm.monitor(torch.nn.Module())
        debugger.gm.monitor.assert_called_once()

    def tearDown(self):
        if os.path.exists("./dump_path/"):
            shutil.rmtree("./dump_path/")
        if os.path.exists("./grad_output/"):
            shutil.rmtree("./grad_output/")


class TestIterTracer(unittest.TestCase):
    def setUp(self):
        self.debugger = MagicMock()
        self.debugger.service.first_start = False
        self.debugger.enable_dataloader = True
        self.ori_instance = PrecisionDebugger._instance
        PrecisionDebugger._instance = self.debugger

    def tearDown(self):
        PrecisionDebugger._instance = self.ori_instance

    def test_debugger_with_not_first_start(self):
        @PrecisionDebugger._iter_tracer
        def test_func():
            return "test case 1"

        result = test_func()

        self.assertEqual(result, "test case 1")
        self.debugger.stop.assert_called_once()
        self.debugger.step.assert_called_once()
        self.debugger.start.assert_called_once()

    def test_debugger_with_first_start(self):
        self.debugger.service.first_start = True

        @PrecisionDebugger._iter_tracer
        def test_func():
            return "test case 2"

        result = test_func()
        self.assertEqual(result, "test case 2")
        self.debugger.stop.assert_not_called()
        self.debugger.step.assert_not_called()
        self.debugger.start.assert_called_once()

    def test_no_debugger_instance(self):
        PrecisionDebugger._instance = None

        @PrecisionDebugger._iter_tracer
        def test_func():
            return "test case 3"

        with self.assertRaises(MsprobeException) as context:
            result = test_func()
            self.assertEqual(result, "test case 3")
        self.assertEqual(context.exception.code, MsprobeException.INTERFACE_USAGE_ERROR)
