# coding=utf-8
import unittest
import os
import torch
from msprobe.pytorch.grad_probe.grad_stat_csv import GradStatCsv
from msprobe.pytorch.grad_probe.grad_monitor import GradientMonitor
from msprobe.core.grad_probe.constant import level_adp

grad_tensor = torch.tensor([[-2, 2], [0.2, 0.3]])


class TestGradCSV(unittest.TestCase):
    def test_level_L0_header(self):
        self.assertEqual(['param_name', 'MD5', 'max', 'min', 'norm', 'shape'], 
                         GradStatCsv.generate_csv_header(level_adp["L0"], [-1, 0, 1]))

    def test_level_L1_header(self):
        self.assertEqual(['param_name', 'max', 'min', 'norm', 'shape'], 
                         GradStatCsv.generate_csv_header(level_adp["L1"], [-1, 0, 1]))

    def test_level_L2_header(self):
        self.assertEqual(['param_name', '(-inf, -1]', '(-1, 0]', '(0, 1]', '(1, inf)', '=0', 'max', 'min', 'norm', 'shape'], 
                         GradStatCsv.generate_csv_header(level_adp["L2"], [-1, 0, 1]))

    def test_level_L0_content(self):
        generated_csv_line = GradStatCsv.generate_csv_line("model.conv2d", level_adp["L0"], grad_tensor, [-1, 0, 1])
        self.assertEqual(['model.conv2d', 'e2863940', 2.0, -2.0, 2.851315498352051, [2, 2]],
                         generated_csv_line)

    def test_level_L1_content(self):
        generated_csv_line = GradStatCsv.generate_csv_line("model.conv2d", level_adp["L1"], grad_tensor, [-1, 0, 1])
        self.assertEqual(['model.conv2d', 2.0, -2.0, 2.851315498352051, [2, 2]],
                         generated_csv_line)

    def test_level_L2_content(self):
        generated_csv_line = GradStatCsv.generate_csv_line("model.conv2d", level_adp["L2"], grad_tensor, [-1, 0, 1])
        self.assertEqual(['model.conv2d', 0.25, 0.0, 0.5, 0.25, 0.0, 2.0, -2.0, 2.851315498352051, [2, 2]],
                         generated_csv_line)
