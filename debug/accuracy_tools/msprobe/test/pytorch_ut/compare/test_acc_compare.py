# coding=utf-8
import unittest
from atat.pytorch.compare.acc_compare import rename_api

class TestUtilsMethods(unittest.TestCase):

    def test_rename_api(self):
        test_name_1 = "Distributed.broadcast.0.forward.input.0"
        expect_name_1 = "Distributed.broadcast.input.0"
        actual_name_1 = rename_api(test_name_1, "forward")
        self.assertEqual(actual_name_1, expect_name_1)
        
        test_name_2 = "Torch.sum.0.backward.output.0"
        expect_name_2 = "Torch.sum.output.0"
        actual_name_2 = rename_api(test_name_2, "backward")
        self.assertEqual(actual_name_2, expect_name_2)
        