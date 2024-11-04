# coding=utf-8
import unittest
from unittest.mock import patch
from multiprocessing import Queue

from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.attl import *
from msprobe.core.common.file_utils import create_directory

class TestATTL(unittest.TestCase):
    
    def setUp(self):
        nfs_path = "temp_nfs_path"
        create_directory(nfs_path)
        self.nfs_path = os.path.realpath(nfs_path)
        self.session_id = "test_session"
        self.session_config = ATTLConfig(is_benchmark_device=False, connect_ip='127.0.0.1', 
                                         connect_port=8080, nfs_path=self.nfs_path , check_sum=False, queue_size=100)
        self.attls = ATTL(self.session_id, self.session_config, need_dump=False)
        self.buffer = ApiData('test_api', args=(torch.randn(2, 2),), kwargs={'device': 'cpu'}, 
                              result=torch.randn(2, 2), step=1, rank=1)
        
    def tearDown(self):
        for filename in os.listdir(self.nfs_path):
            os.remove(os.path.join(self.nfs_path, filename))
        os.rmdir(self.nfs_path)
    
    def test_attl_config(self):
        config = ATTLConfig(is_benchmark_device=True, connect_ip='192.168.1.1', connect_port=9090,
                            nfs_path=self.nfs_path, tls_path='/path/to/tls', check_sum=False, queue_size=100)
        self.assertEqual(config.is_benchmark_device, True)
        self.assertEqual(config.connect_ip, '192.168.1.1')
        self.assertEqual(config.connect_port, 9090)
        self.assertEqual(config.nfs_path, self.nfs_path)
        self.assertEqual(config.tls_path, '/path/to/tls')
        self.assertFalse(config.check_sum)
        self.assertEqual(config.queue_size, 100)

    @patch('msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.attl.move2target_device')
    def test_upload_api_data(self, mock_move2target_device):
        mock_move2target_device.return_value = self.buffer
        self.attls.upload(self.buffer)
        mock_move2target_device.assert_called_once_with(self.buffer, torch.device('cpu'))

    @patch('glob.glob')
    def test_download_no_files(self, mock_glob):
        mock_glob.return_value = []
        result = self.attls.download()
        self.assertIsNone(result)

    @patch('glob.glob')
    @patch('msprobe.pytorch.common.utils.load_pt')
    def test_download_with_exception(self, mock_load_pt, mock_glob):
        mock_glob.return_value = ['/tmp/start_file.pt']
        mock_load_pt.side_effect = Exception('Load error')
        with patch.object(self.attls.logger, 'warning') as mock_logger:
            result = self.attls.download()
            self.assertIsNone(result)
            mock_logger.assert_called_once()

    def test_move2device_exec_tensor(self):
        tensor = torch.randn(2, 2)
        device = torch.device("cpu")
        moved_tensor = move2device_exec(tensor, device)
        self.assertEqual(moved_tensor.device, device)

    def test_move2device_exec_list(self):
        tensor_list = [torch.randn(2, 2), torch.randn(2, 2)]
        device = torch.device("cpu")
        moved_list = move2device_exec(tensor_list, device)
        for tensor in moved_list:
            self.assertEqual(tensor.device, device)

    def test_move2device_exec_tuple(self):
        tensor_tuple = (torch.randn(2, 2), torch.randn(2, 2))
        device = torch.device("cpu")
        moved_tuple = move2device_exec(tensor_tuple, device)
        for tensor in moved_tuple:
            self.assertEqual(tensor.device, device)

    def test_move2device_exec_dict(self):
        tensor_dict = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}
        device = torch.device("cpu")
        moved_dict = move2device_exec(tensor_dict, device)
        for tensor in moved_dict.values():
            self.assertEqual(tensor.device, device)

    def test_move2device_exec_device(self):
        device = torch.device("cpu")
        moved_device = move2device_exec(torch.device("cpu"), device)
        self.assertEqual(moved_device, device)

    def test_move2device_exec_non_tensor(self):
        obj = "This is a string"
        device = torch.device("cpu")
        self.assertEqual(move2device_exec(obj, device), obj)
        
    def test_move2target_device_to_cpu(self):
        tensor_args = (torch.randn(2, 2), torch.randn(3, 3))
        tensor_kwargs = {'key1': torch.randn(2, 2), 'key2': torch.randn(3, 3)}
        tensor_result = torch.randn(2, 2)
        buffer = ApiData('test_api', tensor_args, tensor_kwargs, tensor_result, 1, 1)
        target_device = torch.device('cpu')
        moved_buffer = move2target_device(buffer, target_device)
        self.assertEqual(moved_buffer.result.device, target_device)
        for tensor in moved_buffer.args:
            self.assertEqual(tensor.device, target_device)
        for tensor in moved_buffer.kwargs.values():
            self.assertEqual(tensor.device, target_device)
