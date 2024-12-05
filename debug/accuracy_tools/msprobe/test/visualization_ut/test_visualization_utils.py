import os
import unittest
from msprobe.visualization.utils import (load_json_file, load_data_json_file, str2float, check_directory_content,
                                         GraphConst)


class TestMappingConfig(unittest.TestCase):

    def setUp(self):
        self.yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mapping.yaml")
        self.input = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input")

    def test_load_json_file(self):
        result = load_json_file(self.yaml_path)
        self.assertEqual(result, {})

    def test_load_data_json_file(self):
        result = load_data_json_file(self.yaml_path)
        self.assertEqual(result, {})

    def test_str2float(self):
        result = str2float('23.4%')
        self.assertAlmostEqual(result, 0.234)
        result = str2float('2.3.4%')
        self.assertAlmostEqual(result, 0)

    def test_check_directory_content(self):
        input_type = check_directory_content(self.input)
        self.assertEqual(input_type, GraphConst.STEPS)

        input_type = check_directory_content(os.path.join(self.input, "step0"))
        self.assertEqual(input_type, GraphConst.RANKS)

        with self.assertRaises(ValueError):
            check_directory_content(os.path.join(self.input, "step1"))

        input_type = check_directory_content(os.path.join(self.input, "step0", "rank0"))
        self.assertEqual(input_type, GraphConst.FILES)


if __name__ == '__main__':
    unittest.main()
