import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import torch

from msprobe.core.common.const import Const, MsgConst
from msprobe.core.common.utils import get_real_step_or_rank
from msprobe.core.common.exceptions import MsprobeException, FileCheckException
from msprobe.pytorch.debugger.precision_debugger import PrecisionDebugger, iter_tracer
from msprobe.pytorch.grad_probe.grad_monitor import GradientMonitor
from msprobe.test.pytorch_ut.grad_probe.test_grad_monitor import common_config, task_config


class Args:
    def __init__(self, config_path=None, task=None, dump_path=None, level=None, model=None):
        self.config_path = config_path
        self.task = task
        self.dump_path = dump_path
        self.level = level
        self.model = model


class TestPrecisionDebugger(unittest.TestCase):

    def test_init(self):
        gm = GradientMonitor(common_config, task_config)
        self.assertIsNotNone(gm)
        step = get_real_step_or_rank([0, 1, "3-5"], Const.STEP)
        self.assertListEqual(step, [0, 1, 3, 4, 5])

    def test_instance(self):
        debugger1 = PrecisionDebugger(dump_path="./dump_path")
        debugger2 = PrecisionDebugger(dump_path="./dump_path")
        self.assertIs(debugger1.instance, debugger2.instance)

    def test_check_input_params(self):
        args = Args(config_path = 1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger.check_input_params(args)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(config_path = "./")
        with self.assertRaises(FileCheckException) as context:
            PrecisionDebugger.check_input_params(args)
        self.assertEqual(context.exception.code, FileCheckException.INVALID_FILE_ERROR)

        args = Args(task = 1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger.check_input_params(args)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(dump_path = 1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger.check_input_params(args)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(level = 1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger.check_input_params(args)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(config_path = os.path.join(os.path.dirname(__file__), "../../../config.json"), 
                    task = Const.TASK_LIST[0], 
                    dump_path="./dump_path", 
                    level = Const.LEVEL_LIST[0], 
                    model = torch.nn.Module())
        checked_input_params = PrecisionDebugger.check_input_params(args)
        self.assertIsNone(checked_input_params)

    def test_start_grad_probe(self):
        with self.assertRaises(Exception) as context:
            PrecisionDebugger._instance = None
            PrecisionDebugger.start()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)

        PrecisionDebugger._instance = PrecisionDebugger(task=Const.GRAD_PROBE, dump_path="./dump_path")
        checked_start = PrecisionDebugger.start()
        self.assertIsNone(checked_start)

    def test_start_statistics(self):
        debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.config = MagicMock()
        debugger.model = 'model'
        debugger.api_origin = 'api_origin'
        debugger.task = ''
        debugger.start()
        debugger.service.start.assert_called_once_with('model', 'api_origin')
        self.assertFalse(debugger.api_origin)

    def test_forward_backward_dump_end(self):
        debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.forward_backward_dump_end()
        debugger.service.forward_backward_dump_end.assert_called_once()
        self.assertTrue(debugger.api_origin)

    def test_stop_grad_probe(self):
        with self.assertRaises(Exception) as context:
            PrecisionDebugger._instance = None
            PrecisionDebugger.stop()
        self.assertEqual(str(context.exception), MsgConst.NOT_CREATED_INSTANCE)

        PrecisionDebugger._instance = PrecisionDebugger(task=Const.GRAD_PROBE, dump_path="./dump_path")
        checked_stop = PrecisionDebugger.stop()
        self.assertIsNone(checked_stop)

    def test_stop_statistics(self):
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

        debugger = PrecisionDebugger(task=Const.STATISTICS, dump_path="./dump_path")
        checked_monitor = debugger.monitor(torch.nn.Module())
        self.assertIsNone(checked_monitor)

        debugger = PrecisionDebugger(task=Const.GRAD_PROBE, dump_path="./dump_path")
        debugger.gm = MagicMock()
        debugger.service = MagicMock()
        debugger.task = Const.GRAD_PROBE
        debugger.gm.monitor(torch.nn.Module())
        debugger.gm.monitor.assert_called_once()

    @patch('msprobe.pytorch.debugger.precision_debugger.PrecisionDebugger')
    def test_iter_tracer(self, mock_debugger):
        mock_debugger_instance = mock_debugger.instance = MagicMock()
        mock_debugger_instance.service.first_start = False

        @iter_tracer
        def dataloader_func():
            return "test_iter_tracer"
        result = dataloader_func()
        self.assertEqual(result, "test_iter_tracer")

        mock_debugger_instance.stop.assert_called_once()
        mock_debugger_instance.step.assert_called_once()
        mock_debugger_instance.start.assert_called_once()
        self.assertTrue(mock_debugger_instance.enable_dataloader)

    @patch('msprobe.pytorch.debugger.precision_debugger.PrecisionDebugger')
    def test_iter_tracer_first_start(self, mock_debugger):
        mock_debugger_instance = mock_debugger.instance = MagicMock()
        mock_debugger_instance.service.first_start = True

        @iter_tracer
        def dataloader_func():
            return "test_iter_tracer"
        result = dataloader_func()
        self.assertEqual(result, "test_iter_tracer")

        mock_debugger_instance.stop.assert_not_called()
        mock_debugger_instance.step.assert_not_called()
        mock_debugger_instance.start.assert_called_once()
        self.assertTrue(mock_debugger_instance.enable_dataloader)
    
    def tearDown(self):
        if os.path.exists("./dump_path/"):
            shutil.rmtree("./dump_path/")
        if os.path.exists("./grad_output/"):
            shutil.rmtree("./grad_output/")
