import unittest

import torch

from msprobe.pytorch.monitor.utils import filter_special_chars, MsgConst, get_param_struct, validate_ops, \
    validate_ranks, validate_targets, validate_print_struct, validate_ur_distribution, validate_xy_distribution, \
    validate_mg_distribution, validate_wg_distribution, validate_cc_distribution, validate_alert, validate_config


class TestValidationFunctions(unittest.TestCase):
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
        self.assertEqual(valid_ops, ['norm', 'max'])

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


if __name__ == '__main__':
    unittest.main()
