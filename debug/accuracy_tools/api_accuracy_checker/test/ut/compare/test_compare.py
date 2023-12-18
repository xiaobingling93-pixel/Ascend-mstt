import csv
import os
import shutil
import time
import unittest
import torch.nn.functional

from api_accuracy_checker.compare.compare import Comparator

current_time = time.strftime("%Y%m%d%H%M%S")
RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + '.csv'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.output_path = os.path.join(base_dir, "../compare_result")
        os.mkdir(self.output_path, mode=0o750)
        self.result_csv_path = os.path.join(self.output_path, RESULT_FILE_NAME)
        self.details_csv_path = os.path.join(self.output_path, DETAILS_FILE_NAME)
        self.is_continue_run_ut = False
        self.compare = Comparator(self.result_csv_path, self.details_csv_path, self.is_continue_run_ut)

    def tearDown(self) -> None:
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

    def test_compare_dropout(self):
        dummmy_input = torch.randn(100, 100)
        bench_out = torch.nn.functional.dropout2d(dummmy_input, 0.3)
        npu_out = torch.nn.functional.dropout2d(dummmy_input, 0.3)
        self.assertTrue(self.compare._compare_dropout(bench_out, npu_out))

    def test_compare_core_wrapper(self):
        dummy_input = torch.randn(100, 100)
        bench_out, npu_out = dummy_input, dummy_input
        test_final_success, detailed_result_total = self.compare._compare_core_wrapper(bench_out, npu_out)
        self.assertTrue(test_final_success)
        self.assertEqual(detailed_result_total, [['torch.float32', 'torch.float32', (100, 100), 1.0, 0.0, 'N/A', 'N/A',
                                                  'N/A', 'N/A', 'pass', '\n']])

        bench_out, npu_out = [dummy_input, dummy_input], [dummy_input, dummy_input]
        test_final_success, detailed_result_total = self.compare._compare_core_wrapper(bench_out, npu_out)
        self.assertTrue(test_final_success)
        self.assertEqual(detailed_result_total, [['torch.float32', 'torch.float32', (100, 100), 1.0, 0.0, 'N/A', 'N/A',
                                                  'N/A', 'N/A', 'pass', '\n'], ['torch.float32', 'torch.float32',
                                                                                (100, 100), 1.0, 0.0, 'N/A', 'N/A',
                                                                                'N/A', 'N/A', 'pass', '\n']])

    def test_compare_output(self):
        bench_out, npu_out = torch.randn(100, 100), torch.randn(100, 100)
        bench_grad, npu_grad = [torch.randn(100, 100)], [torch.randn(100, 100)]
        api_name = 'Functional*conv2d*0'
        is_fwd_success, is_bwd_success = self.compare.compare_output(api_name, bench_out, npu_out, bench_grad, npu_grad)
        self.assertFalse(is_fwd_success)
        self.assertFalse(is_bwd_success)

        dummy_input = torch.randn(100, 100)
        bench_out, npu_out = dummy_input, dummy_input
        is_fwd_success, is_bwd_success = self.compare.compare_output(api_name, bench_out, npu_out)
        self.assertTrue(is_fwd_success)
        self.assertTrue(is_bwd_success)

    def test_record_results(self):
        args = ('Functional*conv2d*0', False, 'N/A', [['torch.float64', 'torch.float32', (32, 64, 112, 112), 1.0,
                                                       0.012798667686, 'N/A', 0.81631212311, 0.159979121213, 'N/A',
                                                       'error', '\n']], None)
        self.compare.record_results(*args)
        with open(self.details_csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            api_name_list = [row[0] for row in csv_reader]
        self.assertEqual(api_name_list[0], 'Functional*conv2d*0.forward.output.0')
