import os
import glob
import unittest
import logging
from unittest.mock import patch, mock_open, MagicMock
import json
import signal
from atat.pytorch.api_accuracy_checker.run_ut.multi_run_ut import split_json_file, signal_handler, run_parallel_ut, \
    prepare_config, main, ParallelUTConfig


class TestMultiRunUT(unittest.TestCase):

    def setUp(self):
        self.test_json_file = 'dump.json'
        self.test_data = {'data': {'key1': 'TRUE', 'key2': 'TRUE', 'key3': 'TRUE'}}
        self.test_json_content = json.dumps(self.test_data)
        self.forward_split_files_content = [
            {'key1': 'TRUE', 'key2': 'TRUE'},
            {'key3': 'TRUE', 'key4': 'TRUE'}
        ]

    @patch('atat.pytorch.api_accuracy_checker.run_ut.multi_run_ut.FileOpen')
    def test_split_json_file(self, mock_FileOpen):
        mock_FileOpen.return_value.__enter__.return_value = mock_open(read_data=self.test_json_content).return_value
        num_splits = 2
        split_files, total_items = split_json_file(self.test_json_file, num_splits, False)
        self.assertEqual(len(split_files), num_splits)
        self.assertEqual(total_items, len(self.test_data.get('data')))


    @patch('subprocess.Popen')
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load', side_effect=lambda f: {'key1': 'TRUE', 'key2': 'TRUE'})
    def test_run_parallel_ut(self, mock_json_load, mock_file, mock_exists, mock_popen):
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, 1]
        mock_process.stdout.readline.side_effect = ['[ERROR] Test Error Message\n', '']
        mock_popen.return_value = mock_process

        config = ParallelUTConfig(
            api_files=['test.json'],
            out_path='./',
            num_splits=2,
            save_error_data_flag=True,
            jit_compile_flag=False,
            device_id=[0, 1],
            result_csv_path='result.csv',
            total_items=2,
            real_data_path=None
        )

        mock_file.side_effect = [
            mock_open(read_data=json.dumps(self.forward_split_files_content[0])).return_value,
            mock_open(read_data=json.dumps(self.forward_split_files_content[1])).return_value
        ]

        run_parallel_ut(config)

        mock_popen.assert_called()
        mock_exists.assert_called()

    @patch('os.remove')
    @patch('os.path.realpath', side_effect=lambda x: x)
    @patch('atat.pytorch.api_accuracy_checker.run_ut.multi_run_ut.check_link')
    @patch('atat.pytorch.api_accuracy_checker.run_ut.multi_run_ut.check_file_suffix')
    @patch('atat.pytorch.api_accuracy_checker.run_ut.multi_run_ut.FileChecker')
    @patch('atat.pytorch.api_accuracy_checker.run_ut.multi_run_ut.split_json_file',
           return_value=(['forward_split1.json', 'forward_split2.json'], 2))
    def test_prepare_config(self, mock_split_json_file, mock_FileChecker, mock_check_file_suffix, mock_check_link,
                            mock_realpath, mock_remove):
        mock_FileChecker_instance = MagicMock()
        mock_FileChecker_instance.common_check.return_value = './'
        mock_FileChecker.return_value = mock_FileChecker_instance
        args = MagicMock()
        args.api_info = 'forward.json'
        args.out_path = './'
        args.num_splits = 2
        args.save_error_data = True
        args.jit_compile = False
        args.device_id = [0, 1]
        args.result_csv_path = None
        args.real_data_path = None

        config = prepare_config(args)

        self.assertEqual(config.num_splits, 2)
        self.assertTrue(config.save_error_data_flag)
        self.assertFalse(config.jit_compile_flag)
        self.assertEqual(config.device_id, [0, 1])
        self.assertEqual(config.total_items, 2)


    @patch('argparse.ArgumentParser.parse_args')
    @patch('atat.pytorch.api_accuracy_checker.run_ut.multi_run_ut.prepare_config')
    @patch('atat.pytorch.api_accuracy_checker.run_ut.multi_run_ut.run_parallel_ut')
    def test_main(self, mock_run_parallel_ut, mock_prepare_config, mock_parse_args):
        main()
        mock_parse_args.assert_called()
        mock_prepare_config.assert_called()
        mock_run_parallel_ut.assert_called()

    def tearDown(self):
        current_directory = os.getcwd()
        pattern = os.path.join(current_directory, 'accuracy_checking_*')
        files = glob.glob(pattern)

        for file in files:
            try:
                os.remove(file)
                logging.info(f"Deleted file: {file}")
            except Exception as e:
                logging.error(f"Failed to delete file {file}: {e}")

