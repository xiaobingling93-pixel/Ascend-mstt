import unittest
from unittest.mock import patch

import pandas as pd

from msprobe.core.common.utils import CompareException
from msprobe.core.compare.diff_analyze.first_diff_analyze import FirstDiffAnalyze


class TestFirstDiffAnalyze(unittest.TestCase):
    def setUp(self):
        self.header = ['NPU name', 'L2norm diff',
                       'MaxRelativeErr', 'MinRelativeErr', 'MeanRelativeErr', 'NormRelativeErr']
        self.data = [
            ['Functional.conv2d.0.forward.input.0', 1, '0.0%', '0.0%', '0.0%', '0.0%'],
            ['Functional.conv2d.0.forward.input.1', 1, '99.0%', '99.0%', '99.0%', '99.0%']
        ]
        self.result_df = pd.DataFrame(self.data, columns=self.header)

    @patch('msprobe.core.compare.diff_analyze.first_diff_analyze.thresholds',
           {'compare_metrics': ['MaxRelativeErr', 'NormRelativeErr'], 'MaxRelativeErr': [0.5]})
    def test_single_metric_diff_check_true(self):
        first_diff_analyze = FirstDiffAnalyze()
        result = first_diff_analyze.single_metric_diff_check('MaxRelativeErr', '60.0%')
        self.assertTrue(result)

    @patch('msprobe.core.compare.diff_analyze.first_diff_analyze.thresholds',
           {'compare_metrics': ['MaxRelativeErr', 'NormRelativeErr'], 'MaxRelativeErr': [0.5]})
    def test_single_metric_diff_check_false(self):
        first_diff_analyze = FirstDiffAnalyze()
        result = first_diff_analyze.single_metric_diff_check('MaxRelativeErr', '30.0%')
        self.assertFalse(result)

    @patch('msprobe.core.compare.diff_analyze.first_diff_analyze.thresholds',
           {'compare_metrics': ['MaxRelativeErr', 'NormRelativeErr'], 'NormRelativeErr': [0.5]})
    def test_single_metric_diff_check_miss_threshold(self):
        first_diff_analyze = FirstDiffAnalyze()
        with self.assertRaises(CompareException) as context:
            result = first_diff_analyze.single_metric_diff_check('MaxRelativeErr', '30.0%')
        self.assertEqual(context.exception.code, CompareException.MISSING_THRESHOLD_ERROR)

    @patch('msprobe.core.compare.diff_analyze.first_diff_analyze.thresholds',
           {'compare_metrics': ['MaxRelativeErr', 'NormRelativeErr'], 'MaxRelativeErr': [0.5, 1.0]})
    def test_single_metric_diff_check_wrong_threshold(self):
        first_diff_analyze = FirstDiffAnalyze()
        with self.assertRaises(CompareException) as context:
            result = first_diff_analyze.single_metric_diff_check('MaxRelativeErr', '30.0%')
        self.assertEqual(context.exception.code, CompareException.WRONG_THRESHOLD_ERROR)

    def test_single_api_check_within_threshold(self):
        result_slice = [
            ['Functional.conv2d.0.forward.input.0', 1, '0.0%', '0.0%', '0.0%', '0.0%'],
            ['Functional.conv2d.0.forward.input.1', 1, '0.1%', '0.1%', '0.1%', '0.1%']
        ]
        expected_result = {
            'is_same': True,
            'op_items': [
                {'NPU name': 'Functional.conv2d.0.forward.input.0', 'L2norm diff': 1,
                 'MaxRelativeErr': '0.0%', 'MinRelativeErr': '0.0%',
                 'MeanRelativeErr': '0.0%', 'NormRelativeErr': '0.0%'},
                {'NPU name': 'Functional.conv2d.0.forward.input.1', 'L2norm diff': 1,
                 'MaxRelativeErr': '0.1%', 'MinRelativeErr': '0.1%',
                 'MeanRelativeErr': '0.1%', 'NormRelativeErr': '0.1%'}
            ]
        }
        first_diff_analyze = FirstDiffAnalyze()
        result = first_diff_analyze.single_api_check(result_slice, self.header)
        self.assertEqual(result, expected_result)

    def test_single_api_check_exceed_threshold(self):
        result_slice = [
            ['Functional.conv2d.0.forward.input.0', 1, '88.0%', '88.0%', '88.0%', '88.0%'],
            ['Functional.conv2d.0.forward.input.1', 1, '99.0%', '99.0%', '99.0%', '99.0%']
        ]
        expected_result = {
            'is_same': False,
            'op_items': [
                {'NPU name': 'Functional.conv2d.0.forward.input.0', 'L2norm diff': 1,
                 'MaxRelativeErr': '88.0%', 'MinRelativeErr': '88.0%',
                 'MeanRelativeErr': '88.0%', 'NormRelativeErr': '88.0%'},
                {'NPU name': 'Functional.conv2d.0.forward.input.1', 'L2norm diff': 1,
                 'MaxRelativeErr': '99.0%', 'MinRelativeErr': '99.0%',
                 'MeanRelativeErr': '99.0%', 'NormRelativeErr': '99.0%'},
            ]
        }
        first_diff_analyze = FirstDiffAnalyze()
        result = first_diff_analyze.single_api_check(result_slice, self.header)
        self.assertEqual(result, expected_result)

    def test_check(self):
        expected_result = {
            'Functional.conv2d.0.forward': {
                'is_same': False,
                'op_items': [
                    {'NPU name': 'Functional.conv2d.0.forward.input.0', 'L2norm diff': 1,
                     'MaxRelativeErr': '0.0%', 'MinRelativeErr': '0.0%',
                     'MeanRelativeErr': '0.0%', 'NormRelativeErr': '0.0%'},
                    {'NPU name': 'Functional.conv2d.0.forward.input.1', 'L2norm diff': 1,
                     'MaxRelativeErr': '99.0%', 'MinRelativeErr': '99.0%',
                     'MeanRelativeErr': '99.0%', 'NormRelativeErr': '99.0%'},
                ]
            }
        }
        first_diff_analyze = FirstDiffAnalyze()
        result = first_diff_analyze.check(self.result_df)
        self.assertEqual(result, expected_result)
