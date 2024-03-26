# coding=utf-8
import unittest
import os
import torch
from grad_tool.grad_stat_csv import GradStatCsv
from grad_tool.level_adapter import LevelAdapter


grad_tensor = torch.tensor([[-2, 2], [0.2, 0.3]])


class TestGradCSV(unittest.TestCase):
    def test_level_L0_header(self):
        self.assertEqual(['param_name', 'MD5', 'max', 'min', 'norm', 'shape'], 
                         GradStatCsv.generate_csv_header(level=LevelAdapter.level_adapter("L0"), bounds=[-1, 0, 1]))
    
    def test_level_L1_header(self):
        self.assertEqual(['param_name', 'MD5', '(-inf, -1]', '(-1, 0]', '(0, 1]', '(1, inf)', '=0', 'max', 'min', 'norm', 'shape'], 
                         GradStatCsv.generate_csv_header(level=LevelAdapter.level_adapter("L1"), bounds=[-1, 0, 1]))
        
    def test_level_L2_header(self):
        self.assertEqual(['param_name', 'MD5', 'max', 'min', 'norm', 'shape'], 
                         GradStatCsv.generate_csv_header(level=LevelAdapter.level_adapter("L2"), bounds=[-1, 0, 1]))
    
    def test_level_L3_header(self):
        self.assertEqual(['param_name', 'MD5', '(-inf, -1]', '(-1, 0]', '(0, 1]', '(1, inf)', '=0', 'max', 'min', 'norm', 'shape'], 
                         GradStatCsv.generate_csv_header(level=LevelAdapter.level_adapter("L3"), bounds=[-1, 0, 1]))
    
    def test_level_L0_content(self):
        generated_csv_line = GradStatCsv.generate_csv_line(
                level=LevelAdapter.level_adapter("L0"), 
                param_name="model.conv2d", 
                grad=grad_tensor,
                bounds=[-1, 0, 1])
        self.assertEqual(['model.conv2d', '678a6c7d9d9716682b56fda097d0936c', 2.0, -2.0, 2.851315498352051, [2, 2]],
                         generated_csv_line)
        
    def test_level_L1_content(self):
        generated_csv_line = GradStatCsv.generate_csv_line(
                level=LevelAdapter.level_adapter("L1"), 
                param_name="model.conv2d", 
                grad=grad_tensor,
                bounds=[-1, 0, 1])
        self.assertEqual(['model.conv2d', '678a6c7d9d9716682b56fda097d0936c', 0.25, 0.0, 0.5, 0.25, 0.0, 2.0, -2.0, 2.851315498352051, [2, 2]],
                         generated_csv_line)
        
    def test_level_L2_content(self):
        generated_csv_line = GradStatCsv.generate_csv_line(
                level=LevelAdapter.level_adapter("L2"), 
                param_name="model.conv2d", 
                grad=grad_tensor,
                bounds=[-1, 0, 1])
        self.assertEqual(['model.conv2d', '678a6c7d9d9716682b56fda097d0936c', 2.0, -2.0, 2.851315498352051, [2, 2]],
                         generated_csv_line)
        
    def test_level_L3_content(self):
        generated_csv_line = GradStatCsv.generate_csv_line(
                level=LevelAdapter.level_adapter("L3"), 
                param_name="model.conv2d", 
                grad=grad_tensor,
                bounds=[-1, 0, 1])
        self.assertEqual(['model.conv2d', '678a6c7d9d9716682b56fda097d0936c', 0.25, 0.0, 0.5, 0.25, 0.0, 2.0, -2.0, 2.851315498352051, [2, 2]],
                         generated_csv_line)
