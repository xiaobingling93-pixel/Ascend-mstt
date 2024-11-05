import unittest
from unittest.mock import Mock, patch
import torch
import os
import sys

import shutil

from msprobe.pytorch.online_dispatch.dispatch import PtdbgDispatch
from msprobe.pytorch.online_dispatch.utils import DispatchException


class RunParam:
    def __init__(self, aten_api, aten_api_overload_name):
        self.aten_api = aten_api
        self.aten_api_overload_name = aten_api_overload_name
        self.func_namespace = None


class TestPtdbgDispatch(unittest.TestCase):
    def setUp(self):
        self.dump_path = './dump_path'
        if not os.path.exists(self.dump_path):
            os.mkdir(self.dump_path)
        with patch('msprobe.pytorch.online_dispatch.dispatch.is_npu', new=True), \
            patch('msprobe.pytorch.online_dispatch.dispatch.torch_npu') as mock_torch_npu:
            mock_torch_npu._C = Mock()
            mock_torch_npu._C._npu_getDevice.return_value = 1
            self.PtdbgDispatch = PtdbgDispatch(dump_mode='list', dump_path=self.dump_path, api_list=['relu', 'aefa', 'rsqrt'], debug=True)
        self.yaml_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_ops_config.yaml')

    def tearDown(self):
        if os.path.exists(self.dump_path):
            shutil.rmtree(self.dump_path)

    @patch('msprobe.pytorch.online_dispatch.dispatch.logger.info')
    def test_exit(self, mock_info):
        with self.PtdbgDispatch:
            pass
        
        mock_info.assert_called()
        args = mock_info.call_args[0]
        self.assertTrue(args[0].startswith('Dispatch exit'))

    @patch('torch.ops.aten')
    def test_check_fun_success(self, mock_aten):
        run_param = RunParam('my_api', 'my_overload')
        mock_func = Mock()
        mock_aten.my_api = Mock()
        mock_aten.my_api.my_overload = mock_func
        res = self.PtdbgDispatch.check_fun(mock_func, run_param)

        self.assertTrue(res)
        self.assertEqual(run_param.func_namespace, 'aten')

    @patch('torch.ops.aten')
    def test_check_fun_failure(self, mock_aten):
        run_param = RunParam('invalid_api', 'invalid_overload')

        res = self.PtdbgDispatch.check_fun(None, run_param)

        self.assertFalse(res)
        self.assertIsNone(run_param.func_namespace)

    def test_get_dir_name(self):
        res = self.PtdbgDispatch.get_dir_name('my_tag')

        self.assertIn('msprobe_my_tag_rank', res)

    def test_get_ops(self):
        self.PtdbgDispatch.get_ops(self.yaml_file_path)

        self.assertEqual(self.PtdbgDispatch.aten_ops_blacklist, ['rand'])
        self.assertEqual(self.PtdbgDispatch.npu_adjust_autograd, ['to'])

    def test_filter_dump_api(self):
        self.PtdbgDispatch.filter_dump_api()

        self.assertEqual(self.PtdbgDispatch.dump_api_list, ['relu', 'rsqrt'])

    def test_get_dump_flag(self):
        dump_flag, auto_dump_flag = self.PtdbgDispatch.get_dump_flag('rsqrt')

        self.assertEqual(dump_flag, True)
        self.assertEqual(auto_dump_flag, False)

    def test_check_param_dump_mode(self):
        with self.assertRaises(DispatchException):
            self.PtdbgDispatch.dump_mode = 'awfef'
            self.PtdbgDispatch.check_param()

    def test_check_param_dump_api_list(self):
        with self.assertRaises(DispatchException):
            self.PtdbgDispatch.dump_api_list = 'awfef'
            self.PtdbgDispatch.check_param()

    def test_check_param_debug_flag(self):
        with self.assertRaises(DispatchException):
            self.PtdbgDispatch.debug_flag = 'awfef'
            self.PtdbgDispatch.check_param()

    def test_check_param_process_num(self):
        with self.assertRaises(DispatchException):
            self.PtdbgDispatch.process_num = 'awfef'
            self.PtdbgDispatch.check_param()

    @patch('torch._C._dispatch_tls_set_dispatch_key_excluded')
    def test_enable_autograd(self, mock__dispatch_tls_set_dispatch_key_excluded):
        self.PtdbgDispatch.npu_adjust_autograd.append('to')
        self.PtdbgDispatch.enable_autograd('to')

        mock__dispatch_tls_set_dispatch_key_excluded.assert_called_once()
