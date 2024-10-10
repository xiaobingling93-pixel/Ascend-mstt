

import os
import shutil
import json
import numpy as np
from unittest import TestCase, mock
from mindspore import Tensor
from msprobe.mindspore.grad_probe.global_analyzer import CSVGenerator
from msprobe.core.grad_probe.constant import GradConst

class TestGradAnalyzer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_path = "./test_output"
        cls.dump_dir = f"{cls.output_path}/rank0/Dump"
        cls.save_dir = f"{cls.output_path}/rank0"
        os.makedirs(cls.dump_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.output_path):
            shutil.rmtree(cls.output_path)

    def setUp(self):
        # Initialize CSVGenerator instance before each test
        self.csv_generator = CSVGenerator()
        self.csv_generator.init(mock.Mock(**{
            'get_context.side_effect': lambda x: {
                GradConst.OUTPUT_PATH: self.output_path,
                GradConst.LEVEL: GradConst.LEVEL2,
                GradConst.BOUNDS: [-0.1, 0.0, 0.1]
            }[x]
        }))

    def test_run_with_no_dump_dir(self):
        # Test run method when dump directory does not exist
        shutil.rmtree(self.dump_dir)
        with mock.patch("time.sleep", return_value=None):
            with self.assertRaises(SystemExit):
                self.csv_generator.run()

    def test_traverse_files_with_empty_directory(self):
        # Test traverse_files method with empty dump directory
        self.csv_generator.traverse_files([])
        self.assertEqual(len(os.listdir(self.dump_dir)), 0)

    def test_load_npy_data(self):
        # Test load_npy_data method with mock data
        test_file_path = os.path.join(self.dump_dir, "test_data.npy")
        np.save(test_file_path, np.array([1, 2, 3]))
        data = self.csv_generator.load_npy_data(test_file_path)
        self.assertIsNotNone(data)
        np.testing.assert_array_equal(data, np.array([1, 2, 3]))

    def test_create_csv_file(self):
        # Test CSV file creation
        self.csv_generator.current_step = 0
        self.csv_generator.create_csv_file()
        csv_path = os.path.join(self.save_dir, "grad_summary_0.csv")
        self.assertTrue(os.path.exists(csv_path))

    def test_check_valid_data(self):
        # Test check_valid method with valid and invalid data
        valid_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        invalid_data = np.array([0, 1, 2, 3, 4, 5])
        self.assertTrue(self.csv_generator.check_valid(valid_data))
        self.assertFalse(self.csv_generator.check_valid(invalid_data))

    def test_gen_csv_line(self):
        # Test gen_csv_line method
        file_path = os.path.join(self.dump_dir, "0_test_param.npy")
        stat_data = np.array([0, 1, 2, 3, 4, 2, 2, 2, 1, 0])
        with mock.patch.object(self.csv_generator.cache_list, 'append') as mock_append:
            self.csv_generator.gen_csv_line(file_path, stat_data)
            mock_append.assert_called_once()

if __name__ == "__main__":
    from unittest import main
    main()