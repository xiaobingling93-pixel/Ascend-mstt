import os
import shutil
import json
import time
import numpy as np
import mindspore as ms
from unittest import TestCase, mock
from unittest.mock import patch
from mindspore import Tensor, Parameter
from msprobe.mindspore.grad_probe.grad_analyzer import CSVGenerator, grad_dump, GradDumpConfig
from msprobe.mindspore.grad_probe.global_context import grad_context
from msprobe.core.grad_probe.constant import GradConst


class TestGradAnalyzer(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_path = "./test_output"
        cls.time_stamp = str(int(time.time()))
        cls.dump_dir = f"{cls.output_path}/rank0/Dump{cls.time_stamp}"
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
                GradConst.BOUNDS: [-0.1, 0.0, 0.1],
                GradConst.TIME_STAMP: self.time_stamp,
            }[x]
        }))
        # Clear dump directory before each test
        shutil.rmtree(self.dump_dir, ignore_errors=True)
        os.makedirs(self.dump_dir, exist_ok=True)

    def test_run_with_no_dump_dir(self):
        # Test run method when dump directory does not exist
        shutil.rmtree(self.dump_dir)
        with mock.patch("time.sleep", side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):
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
        invalid_data_long = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        invalid_data = np.array([0, 1, 2, 3, 4, 5])
        valid_data = np.array([0, 0, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0])
        # with patch('grad_context.get_context', return_value=GradConst.LEVEL2)
        self.assertFalse(self.csv_generator.check_valid(invalid_data_long))
        self.assertFalse(self.csv_generator.check_valid(invalid_data))

        # in level2, valid case should be: 4th position is shape_dim, {4+shape_dim+1}th position is dist_dim, length equals shape_dim+dist_dim+7
        with mock.patch.object(grad_context, 'get_context', return_value=GradConst.LEVEL2):
            self.assertTrue(self.csv_generator.check_valid(valid_data))

    def test_gen_csv_line(self):
        # Test gen_csv_line method
        file_path = os.path.join(self.dump_dir, "0_test_param.npy")
        stat_data = np.array([0, 1, 2, 3, 4, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0])
        with mock.patch.object(self.csv_generator.cache_list, 'append') as mock_append:
            self.csv_generator.gen_csv_line(file_path, stat_data)
            mock_append.assert_called_once()

        file_path = os.path.join(self.dump_dir, "0.npy")
        with mock.patch.object(self.csv_generator.cache_list, 'append') as mock_append:
            with self.assertRaises(RuntimeError):
                self.csv_generator.gen_csv_line(file_path, stat_data)

    def test_grad_dump(self):
        # Test grad_dump function with numpy file output
        dump_dir = self.dump_dir
        g_name = "test_grad"
        dump_step = Parameter(Tensor([1.0]), name="dump_step")
        grad = Tensor(np.array([0.1, 0.2, 0.3]), dtype=ms.float32)
        level = GradConst.LEVEL2
        bounds = [-0.1, 0.0, 0.1]

        # Run the grad_dump function
        try:
            conf = GradDumpConfig(dump_dir=dump_dir, g_name=g_name, dump_step=dump_step, grad=grad, level=level,
                                  bounds=bounds)
            grad_dump(conf)
        except RuntimeError as e:
            # If TensorDump fails due to environment, skip the file existence check
            self.skipTest(f"TensorDump operation failed: {e}")

        # Verify if the expected dump files are created with the correct naming convention
        expected_files = ["0_test_grad.npy", "1_test_grad_dir.npy"]
        for expected_file in expected_files:
            expected_path = os.path.join(dump_dir, expected_file)
            self.assertTrue(os.path.exists(expected_path), f"Expected file {expected_file} does not exist.")

        # Load the saved numpy arrays and check their contents
        expected_grad_content = np.array([1.0, 0.3, 0.1, 0.37416577, 1.0, 3.0, 5.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0])
        expected_grad_dir_content = np.array([True, True, True])

        for i, expected_file in enumerate(expected_files):
            file_path = os.path.join(dump_dir, expected_file)
            data = np.load(file_path)
            print(f"Content of {file_path}: {data}")
            # Add assertions for the contents
            if i == 0:
                # Check the contents of "0_test_grad.npy"
                np.testing.assert_allclose(data, expected_grad_content, rtol=1e-5,
                                           err_msg=f"Content of {expected_file} does not match expected values.")
            elif i == 1:
                # Check the contents of "1_test_grad_dir.npy"
                np.testing.assert_array_equal(data, expected_grad_dir_content,
                                              err_msg=f"Content of {expected_file} does not match expected values.")

    def test_stop_method(self):
        # Test stop method to ensure stop_event is set
        self.csv_generator.stop()
        self.assertTrue(self.csv_generator.stop_event.is_set())

    def test_traverse_files_with_data(self):
        # Test traverse_files method with mock npy files
        npy_files = ["0_test_param.npy"]
        test_file_path = os.path.join(self.dump_dir, "0_test_param.npy")
        np.save(test_file_path, np.array([1, 2, 3, 4, 5]))
        with mock.patch.object(self.csv_generator, 'load_npy_data', return_value=np.array([1, 2, 3, 4, 5])):
            self.csv_generator.traverse_files(npy_files)
        self.assertFalse(os.path.exists(test_file_path))

        npy_files = ["step_finish.npy"]
        test_file_path = os.path.join(self.dump_dir, "step_finish.npy")
        np.save(test_file_path, np.array([1, 2, 3, 4, 5]))
        with mock.patch.object(self.csv_generator, 'load_npy_data', return_value=np.array([1, 2, 3, 4, 5])):
            self.csv_generator.traverse_files(npy_files)
            self.assertTrue(self.csv_generator.last_finish)

        npy_files = ["step_dir.npy"]
        test_file_path = os.path.join(self.dump_dir, "step_dir.npy")
        np.save(test_file_path, np.array([1, 2, 3, 4, 5]))
        with mock.patch.object(self.csv_generator, 'load_npy_data', return_value=np.array([1, 2, 3, 4, 5])):
            with self.assertRaises(RuntimeError):
                self.csv_generator.traverse_files(npy_files)

    def test_traverse_files_with_data_successful_move(self):
        npy_files = ["step_dir.npy"]
        self.csv_generator.current_step = 0
        test_file_path = os.path.join(self.dump_dir, "step_dir.npy")
        np.save(test_file_path, np.array([1, 2, 3, 4, 5]))
        with mock.patch.object(self.csv_generator, 'load_npy_data', return_value=np.array([1, 2, 3, 4, 5])):
            self.csv_generator.traverse_files(npy_files)
            dst_file_path = os.path.join(self.save_dir, f"step{self.csv_generator.current_step}", "dir.npy")
            assert os.path.exists(dst_file_path)
            real_tensor = np.load(dst_file_path)
            self.assertTrue((real_tensor == np.array([1, 2, 3, 4, 5])).all())


if __name__ == "__main__":
    from unittest import main

    main()
