import unittest
from msprobe.core.compare.layer_mapping.layer_mapping import generate_index_set


class TestLayerMapping(unittest.TestCase):

    def test_generate_index_set_then_pass(self):
        data0 = [0, 0]
        data1 = [[0, 0, 0]]
        data2 = [[0, 0, 0], 0]
        data3 = [[0, 0, 0], 0, [0], [0,[0,[0]]]]
        data4 = []
        expect0 = {"0", "1"}
        expect1 = {"0.0", "0.1", "0.2"}
        expect2 = {"0.0", "0.1", "0.2", "1"}
        expect3 = {"0.0", "0.1", "0.2", "1", "2.0", "3.0", "3.1.0", "3.1.1.0"}
        expect4 = set()
        res0 = generate_index_set(data0)
        self.assertEqual(res0, expect0)
        res1 = generate_index_set(data1)
        self.assertEqual(res1, expect1)
        res2 = generate_index_set(data2)
        self.assertEqual(res2, expect2)
        res3 = generate_index_set(data3)
        self.assertEqual(res3, expect3)
        res4 = generate_index_set(data4)
        self.assertEqual(res4, expect4)
