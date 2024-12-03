import os.path
import shutil
import time
import unittest
from unittest.mock import patch

import pandas as pd

from msprobe.pytorch.monitor.module_hook import CommunicationContext, GradContext, ModuleHookContext, \
    param_is_not_tensor_parallel_duplicate, param_is_data_parallel_duplicate
from msprobe.test.pytorch_ut.monitor.demo_model import monitor_demo

base_dir = os.path.dirname(os.path.realpath(__file__))


def clean_output(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def get_new_item(list1, list2):
    """get list1 - list2"""
    return [x for x in list1 if x not in list2]


class TestModuleHook(unittest.TestCase):
    monitor_output = "./monitor_output"

    def setUp(self) -> None:
        clean_output(self.monitor_output)

    def tearDown(self) -> None:
        clean_output(self.monitor_output)

    def test_print_struct(self):
        print_struct_config = os.path.join(base_dir, "config/struct_config.json")
        with self.assertRaises(Exception) as context:
            monitor_demo(print_struct_config)
        self.assertEqual(str(context.exception), "exit after first step when print model struct")

    def test_xy_wg_mv_ur(self):
        output_list = []

        # # test_xy_distribution
        xy_config = os.path.join(base_dir, "config/xy_config.json")
        monitor_demo(xy_config)
        # validate output file
        csv_dir = get_new_item(os.listdir(self.monitor_output), output_list)
        output_list.append(csv_dir[0])
        self.assertEqual(len(csv_dir), 1)
        actv_0_csv = os.path.join(self.monitor_output, csv_dir[0], "actv_0-0.csv")
        actv_grad_0_csv = os.path.join(self.monitor_output, csv_dir[0], "actv_grad_0-0.csv")
        self.assertTrue(os.path.exists(actv_0_csv))
        self.assertTrue(os.path.exists(actv_grad_0_csv))
        # validate columns and lines
        actv_0 = pd.read_csv(actv_0_csv)
        expect_columns = ['vpp_stage', 'module_name', 'step', 'input.norm', 'output.norm']
        self.assertListEqual(list(actv_0.columns), expect_columns)
        self.assertEqual(actv_0.shape, tuple([3, 5]))
        actv_grad_0 = pd.read_csv(actv_grad_0_csv)
        expect_columns = ['vpp_stage', 'module_name', 'step', 'input_grad.norm', 'output_grad.norm']
        self.assertListEqual(list(actv_grad_0.columns), expect_columns)
        self.assertEqual(actv_0.shape, tuple([3, 5]))
        time.sleep(20)

        # # test_wg_distribution
        wg_config = os.path.join(base_dir, "config/wg_config.json")
        monitor_demo(wg_config)
        # validate output file
        csv_dir = get_new_item(os.listdir(self.monitor_output), output_list)
        output_list.append(csv_dir[0])
        self.assertEqual(len(csv_dir), 1)
        grad_reduced_0_csv = os.path.join(self.monitor_output, csv_dir[0], "grad_reduced_0-0.csv")
        grad_unreduced_0_csv = os.path.join(self.monitor_output, csv_dir[0], "grad_unreduced_0-0.csv")
        self.assertTrue(os.path.exists(grad_reduced_0_csv))
        self.assertTrue(os.path.exists(grad_unreduced_0_csv))
        # validate columns and lines
        expect_columns = ["vpp_stage", "param_name", "step", "norm"]
        grad_reduced_0 = pd.read_csv(grad_reduced_0_csv)
        self.assertListEqual(list(grad_reduced_0.columns), expect_columns)
        self.assertEqual(grad_reduced_0.shape, tuple([2, 4]))
        grad_unreduced_0 = pd.read_csv(grad_unreduced_0_csv)
        self.assertListEqual(list(grad_unreduced_0.columns), expect_columns)
        self.assertEqual(grad_unreduced_0.shape, tuple([2, 4]))
        time.sleep(20)

        # # test_mv_distribution
        mv_config = os.path.join(base_dir, "config/mv_config.json")
        monitor_demo(mv_config)
        # validate output file
        csv_dir = get_new_item(os.listdir(self.monitor_output), output_list)
        output_list.append(csv_dir[0])
        self.assertEqual(len(csv_dir), 1)
        exp_avg_1_csv = os.path.join(self.monitor_output, csv_dir[0], "exp_avg_1-1.csv")
        exp_avg_sq_1_csv = os.path.join(self.monitor_output, csv_dir[0], "exp_avg_sq_1-1.csv")
        self.assertTrue(os.path.exists(exp_avg_1_csv))
        self.assertTrue(os.path.exists(exp_avg_sq_1_csv))
        # validate columns and lines
        expect_columns = ["vpp_stage", "param_name", "step", "norm"]
        exp_avg_1 = pd.read_csv(exp_avg_1_csv)
        self.assertListEqual(list(exp_avg_1.columns), expect_columns)
        self.assertEqual(exp_avg_1.shape, tuple([2, 4]))
        exp_avg_sq_1 = pd.read_csv(exp_avg_sq_1_csv)
        self.assertListEqual(list(exp_avg_sq_1.columns), expect_columns)
        self.assertEqual(exp_avg_sq_1.shape, tuple([2, 4]))
        time.sleep(20)

        # # test_ur_distribution
        ur_config = os.path.join(base_dir, "config/ur_config.json")
        monitor_demo(ur_config)
        # validate output file
        csv_dir = get_new_item(os.listdir(self.monitor_output), output_list)
        output_list.append(csv_dir[0])
        self.assertEqual(len(csv_dir), 1)
        tb_dir = os.listdir(os.path.join(self.monitor_output, csv_dir[0]))
        self.assertEqual(len(tb_dir), 1)
        self.assertTrue(tb_dir[0].startswith("events.out.tfevents."))


class TestParamIsNotTensorParallelDuplicate(unittest.TestCase):
    @patch('torch.distributed.get_rank')
    def test_param_is_not_tensor_parallel_duplicate(self, mock_get_rank):
        class MockParam:
            def __init__(self, tensor_model_parallel):
                self.tensor_model_parallel = tensor_model_parallel

        param = MockParam(True)
        tp_group = 'dummy_group'
        self.assertTrue(param_is_not_tensor_parallel_duplicate(param, tp_group))


class TestParamIsDataParallelDuplicate(unittest.TestCase):
    @patch('torch.distributed.get_rank')
    def test_param_is_data_parallel_duplicate_true(self, mock_get_rank):
        mock_get_rank.return_value = 1
        dp_group = 'dp_group'
        result = param_is_data_parallel_duplicate(dp_group)
        self.assertTrue(result)

    @patch('torch.distributed.get_rank')
    def test_param_is_data_parallel_duplicate_false(self, mock_get_rank):
        mock_get_rank.return_value = 0
        dp_group = 'dp_group'
        result = param_is_data_parallel_duplicate(dp_group)
        self.assertFalse(result)


class TestModuleHookContext(unittest.TestCase):
    def setUp(self):
        self.module_hook_context = ModuleHookContext("test_module")
        self.target_config1 = {'test_module': {'input': {'config': 'input_config'}}}
        self.target_config2 = {'test_module': {'input_grad': 'input_grad_config'}}

    def test_set_format_by_arg_with_dict_value(self):
        self.module_hook_context.set_format_by_arg('input', self.target_config1)
        self.assertEqual(self.module_hook_context.format_by_arg['input'], 'input_config')

    def test_set_format_by_arg_with_non_dict_value(self):
        self.module_hook_context.set_format_by_arg('input_grad', self.target_config2)
        self.assertEqual(self.module_hook_context.format_by_arg['input_grad'], 'input_grad_config')

    def test_set_format_by_arg_with_key_not_in_target_config(self):
        self.module_hook_context.set_format_by_arg('output', self.target_config1)
        self.assertNotIn('output', self.module_hook_context.format_by_arg)

    def test_set_format_by_arg_with_ignore_in(self):
        self.module_hook_context.set_format_by_arg('input', self.target_config2)
        self.assertTrue(self.module_hook_context.ignore_in)


class TestContext(unittest.TestCase):
    def test_communication_context(self):
        cc_ctx = CommunicationContext()
        cc_ctx.reset()
        cc_ctx.data = {'tag1': {'min': [1, 2, 3], 'max': [10, 11, 12]},
                       'tag2': {'min': [16, 17, 18], 'max': [22, 23, 24]}}
        cc_ctx.aggregate()
        expected_aggregated_data = {'tag1': {'max': 12, 'min': 1}, 'tag2': {'max': 24, 'min': 16}}
        self.assertEqual(cc_ctx.data, expected_aggregated_data)

    def test_grad_context(self):
        grad_ctx = GradContext()
        grad_ctx.reset()
        self.assertEqual(grad_ctx.pre, {})
        self.assertEqual(grad_ctx.post, {})


if __name__ == '__main__':
    unittest.main()
