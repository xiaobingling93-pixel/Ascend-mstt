import os
import unittest
from unittest.mock import patch, MagicMock

import torch
from msprobe.core.common.const import MonitorConst

from msprobe.core.monitor.utils import filter_special_chars, MsgConst, validate_ops, validate_ranks, \
    validate_targets, validate_print_struct, validate_ur_distribution, validate_xy_distribution, \
    validate_mg_distribution, validate_wg_distribution, validate_cc_distribution, validate_alert, validate_config, \
    get_output_base_dir
from msprobe.pytorch.monitor.utils import get_param_struct
from msprobe.pytorch.common.utils import is_recomputation



class TestValidationFunctions(unittest.TestCase):

    def test_get_output_base_dir(self):
        # not set env
        if os.getenv(MonitorConst.MONITOR_OUTPUT_DIR):
            del os.environ[MonitorConst.MONITOR_OUTPUT_DIR]
        output_base_dir = get_output_base_dir()
        expect_output_base_dir = "./monitor_output"
        self.assertEqual(output_base_dir, expect_output_base_dir)

        # set env
        os.environ[MonitorConst.MONITOR_OUTPUT_DIR] = "test123"
        output_base_dir = get_output_base_dir()
        expect_output_base_dir = "test123"
        self.assertEqual(output_base_dir, expect_output_base_dir)

    def test_filter_special_chars(self):
        @filter_special_chars
        def func(msg):
            return msg

        self.assertEqual(func(MsgConst.SPECIAL_CHAR[0]), '_')

    def test_get_param_struct(self):
        param = (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
        res = get_param_struct(param)
        self.assertEqual(res['config'], 'tuple[2]')

    def test_validate_ops(self):
        ops = ['op1', 'op2', 'norm', 'max']
        valid_ops = validate_ops(ops)
        self.assertEqual(valid_ops, ['norm', 'max', "shape", "dtype"])

    def test_no_valid_ops(self):
        ops = ['op1', 'op2']
        valid_ops = validate_ops(ops)
        target_ops = [MonitorConst.OP_LIST[0], "shape", "dtype"]
        self.assertEqual(valid_ops, target_ops)

    def test_validate_ranks(self):
        ranks = [0, 1, 2, 3]
        res = validate_ranks(ranks)
        self.assertIsNone(res)

    def test_validate_targets(self):
        targets = {'module_name': {'input': 'tensor'}}
        validate_targets(targets)

    def test_validate_print_struct(self):
        print_struct = True
        validate_print_struct(print_struct)

    def test_validate_ur_distribution(self):
        ur_distribution = True
        validate_ur_distribution(ur_distribution)

    def test_validate_xy_distribution(self):
        xy_distribution = True
        validate_xy_distribution(xy_distribution)

    def test_validate_wg_distribution(self):
        wg_distribution = True
        validate_wg_distribution(wg_distribution)

    def test_validate_mg_distribution(self):
        mg_distribution = True
        validate_mg_distribution(mg_distribution)

    def test_validate_cc_distribution(self):
        cc_distribution = {'enable': True, 'cc_codeline': ['line1'], 'cc_pre_hook': False, 'cc_log_only': True}
        validate_cc_distribution(cc_distribution)

    def test_validate_alert(self):
        alert = {'rules': [{'rule_name': 'AnomalyTurbulence', 'args': {'threshold': 10.0}}], 'dump': True}
        validate_alert(alert)

    def test_validate_config(self):
        config = {
            'ops': ['op1', 'op2'],
            'eps': 1e-8,
            'module_ranks': [0, 1, 2, 3],
            'targets': {'module_name': {'input': 'tensor'}},
            'print_struct': True,
            'ur_distribution': True,
            'xy_distribution': True,
            'wg_distribution': True,
            'mg_distribution': True,
            'cc_distribution': {'enable': True, 'cc_codeline': ['line1'], 'cc_pre_hook': False, 'cc_log_only': True},
            'alert': {'rules': [{'rule_name': 'AnomalyTurbulence', 'args': {'threshold': 10.0}}], 'dump': True}
        }
        validate_config(config)
        target_ops = [MonitorConst.OP_LIST[0], "shape", "dtype"]
        self.assertEqual(config["ops"], target_ops)
        del config["targets"]
        validate_config(config)
        self.assertEqual(config["targets"], {"": {}})
        self.assertEqual(config["all_xy"], True)


class TestIsRecomputation(unittest.TestCase):
    @patch('inspect.stack')
    def test_in_recomputation_megatron(self, mock_stack):
        # 模拟megatron框架下的调用栈
        frame1 = MagicMock()
        frame1.function = 'backward'
        frame1.filename = 'torch/_tensor.py'

        frame2 = MagicMock()
        frame2.function = 'some_function'
        frame2.filename = 'torch/autograd/function.py'

        mock_stack.return_value = [frame1, frame2]

        self.assertTrue(is_recomputation())

    @patch('inspect.stack')
    def test_in_recomputation_mindspeed_L0L1(self, mock_stack):
        # 模拟mindspeed L0&L1场景下的调用栈
        frame1 = MagicMock()
        frame1.function = 'checkpoint_function_backward'
        frame1.filename = 'some_module.py'

        frame2 = MagicMock()
        frame2.function = 'some_other_function'
        frame2.filename = 'torch/autograd/function.py'

        mock_stack.return_value = [frame1, frame2]

        self.assertTrue(is_recomputation())

    @patch('inspect.stack')
    def test_in_recomputation_mindspeed_L2(self, mock_stack):
        # 模拟mindspeed L2场景下的调用栈
        frame1 = MagicMock()
        frame1.function = 'checkpoint_function_backward'
        frame1.filename = 'another_module.py'

        frame2 = MagicMock()
        frame2.function = 'yet_another_function'
        frame2.filename = 'some_file.py'

        frame3 = MagicMock()
        frame3.function = 'final_function'
        frame3.filename = 'torch/autograd/function.py'

        mock_stack.return_value = [frame1, frame2, frame3]

        self.assertTrue(is_recomputation())

    @patch('inspect.stack')
    def test_not_in_recomputation(self, mock_stack):
        # 模拟非重计算阶段的调用栈
        frame1 = MagicMock()
        frame1.function = 'forward'
        frame1.filename = 'my_model.py'

        mock_stack.return_value = [frame1]

        self.assertFalse(is_recomputation())


if __name__ == '__main__':
    unittest.main()
