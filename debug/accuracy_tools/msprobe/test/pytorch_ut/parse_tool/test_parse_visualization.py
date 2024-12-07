import unittest
from unittest.mock import patch
import numpy as np
import os

from msprobe.pytorch.parse_tool.lib.visualization import Visualization


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.visualization = Visualization()
        self.npy_dir = './data.npy'
        var = np.array([1, 2, 3, 4, 5])
        np.save(self.npy_dir, var)
        self.txt_dir = './data.npy.txt'

    def tearDown(self):
        if os.path.exists(self.npy_dir):
            os.remove(self.npy_dir)
        if os.path.exists(self.txt_dir):
            os.remove(self.txt_dir)

    def test_print_npy_summary(self):
        self.visualization.print_npy_summary(self.npy_dir)

        self.assertTrue(os.path.exists(self.txt_dir))

    @patch('msprobe.pytorch.parse_tool.lib.visualization.Visualization.print_npy_summary')
    def test_print_npy_data(self, mock_print_npy_summary):
        with patch('msprobe.pytorch.parse_tool.lib.visualization.Util.check_path_valid', return_value=True), \
            patch('msprobe.pytorch.parse_tool.lib.visualization.Util.check_path_valid', return_value=None):
            self.visualization.print_npy_data(self.npy_dir)

            mock_print_npy_summary.assert_called_once()
