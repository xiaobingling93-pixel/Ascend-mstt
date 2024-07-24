# coding=utf-8
import unittest
from msprobe.pytorch.compare import match


class TestMatch(unittest.TestCase):
    def test_graph_mapping(self):
        op1 = "Aten_convolution_1_forward_0.input.0"
        op2 = "Torch_conv2d_0_forward_0.input.0"
        op3 = "Torch_batch_norm_0_forward_0.input.0"
        op4 = "Aten_convolution.default_1_forward_0.input.0"
        op5 = "Aten_foo_1_forward_0.input.0"
        self.assertTrue(match.graph_mapping.match(op1, op2))
        self.assertTrue(match.graph_mapping.match(op2, op1))
        self.assertTrue(match.graph_mapping.match(op4, op2))
        self.assertTrue(match.graph_mapping.match(op2, op4))
        self.assertFalse(match.graph_mapping.match(op1, op3))
        self.assertFalse(match.graph_mapping.match(op3, op1))
        self.assertFalse(match.graph_mapping.match(op5, op2))
        self.assertFalse(match.graph_mapping.match(op2, op5))
