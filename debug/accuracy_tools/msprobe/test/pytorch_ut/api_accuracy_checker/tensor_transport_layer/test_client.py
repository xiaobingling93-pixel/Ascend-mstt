# coding=utf-8
import unittest
from unittest.mock import patch, MagicMock
from multiprocessing import Queue

from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.client import *
from msprobe.core.common.file_utils import create_directory


class TestClient(unittest.TestCase):
    
    def setUp(self) -> None:
        self.host = "localhost"
        self.port = 8000
        self.check_sum = False
        tls_path = "temp_tls_path"
        create_directory(tls_path)
        self.tls_path = os.path.realpath(tls_path)

    def tearDown(self) -> None:
        for filename in os.listdir(self.tls_path):
            os.remove(os.path.join(self.tls_path, filename))
        os.rmdir(self.tls_path)
    
    def test_TCPDataItem(self):
        data_item = TCPDataItem(data="example_data", sequence_number=10, rank=1, step=2)
        self.assertEqual(data_item.raw_data, "example_data")
        self.assertEqual(data_item.sequence_number, 10)
        self.assertEqual(data_item.rank, 1)
        self.assertEqual(data_item.step, 2)
        self.assertEqual(data_item.retry_times, 0)
        self.assertEqual(data_item.pending_time, 0)
        self.assertEqual(data_item.busy_time, 0)
