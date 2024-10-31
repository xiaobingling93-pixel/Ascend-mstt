# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import queue
import struct
import time
import unittest
from unittest.mock import MagicMock, patch

from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.server import (
    TCPServer,
    ServerProtocol,
    MessageServerFactory
)


class TestTCPServer(unittest.TestCase):
    def setUp(self):
        self.shared_queue = queue.Queue()
        self.tcp_server = TCPServer("6000", self.shared_queue)
        self.tcp_server.tls_path = "/test/path"
        self.tcp_server.factory = MagicMock()

    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.server.reactor")
    def test_run_reactor(self, mock_reactor):
        self.tcp_server.run_reactor()
        mock_reactor.run.assert_called_once_with(installSignalHandlers=False)

    @patch("os.path.exists")
    def test_check_tls_path(self, mock_path_exists):
        mock_path_exists.side_effect = lambda path: True
        server_key, server_crt = self.tcp_server.check_tls_path()

        self.assertEqual(server_key, "/test/path/server.key")
        self.assertEqual(server_crt, "/test/path/server.crt")

    @patch("os.path.exists")
    def test_check_tls_path_missing_key(self, mock_path_exists):
        def side_effect(path):
            if "server.key" in path:
                return False
            return True

        mock_path_exists.side_effect = side_effect

        with self.assertRaises(Exception) as context:
            self.tcp_server.check_tls_path()
        self.assertIn("/test/path/server.key is not exists", str(context.exception))

    def test_is_running(self):
        self.tcp_server.is_running()
        self.tcp_server.factory.is_all_connection_closed.assert_called_once_with()

    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.server.reactor")
    def test_stop(self, mock_reactor):
        self.tcp_server.reactor_thread = MagicMock()
        self.tcp_server.stop()
        mock_reactor.callFromThread.assert_called_once()
        self.tcp_server.reactor_thread.join.assert_called_once()


class TestServerProtocol(unittest.TestCase):
    def setUp(self):
        self.shared_queue = queue.Queue()
        self.server_protocol = ServerProtocol(self.shared_queue)
        self.server_protocol.start_time = time.time()
        self.server_protocol.factory = MagicMock()
        self.server_protocol.factory.transport_dict = {}
        self.server_protocol.factory.transport_list = []
        self.server_protocol.transport = MagicMock()

    def test_connectionMade(self):
        self.server_protocol.connectionMade()
        self.assertEqual(self.server_protocol.tell, 0)
        self.assertEqual(self.server_protocol.factory.transport_dict[self.server_protocol.transport], 1)
        self.assertTrue(self.server_protocol.transport in self.server_protocol.factory.transport_list)

    def test_connectionLost(self):
        self.server_protocol.factory.transport_dict[self.server_protocol.transport] = 1
        self.server_protocol.connectionLost("test")
        self.assertEqual(len(self.server_protocol.factory.transport_dict), 0)
        self.assertEqual(self.server_protocol.consumer_queue.get(), self.server_protocol.ACK_KILL_PROCESS)

    def test_send_ack(self):
        self.server_protocol.sequence_number = 1
        self.server_protocol.rank = 0
        self.server_protocol.step = 0
        self.server_protocol.send_ack(b'test message')
        expected_value = b''.join([
            b'test message',
            b'\x00\x00\x00\x00\x00\x00\x00\x01',
            b'\x00\x00\x00\x00\x00\x00\x00\x00',
            b'\x00\x00\x00\x00\x00\x00\x00\x00',
        ])
        self.server_protocol.transport.write.called_once_with(expected_value)

    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.server.hashlib.md5")
    def test_post_process_error(self, mock_hashlib_md5):
        self.shared_queue.maxsize = 1
        self.server_protocol.send_ack = MagicMock()

        def mock_send_ack_method1():
            self.server_protocol.consumer_queue.put(1)

        def mock_send_ack_method2():
            pass

        self.server_protocol.send_ack.side_effect = [mock_send_ack_method1, mock_send_ack_method2]
        self.server_protocol.check_sum = True
        mock_hashlib_md5.hexdiges.return_value = "123"
        self.server_protocol.rank = 0
        self.server_protocol.step = 0
        self.server_protocol.post_process()
        mock_hashlib_md5.assert_called()
        self.server_protocol.send_ack.assert_any_call(self.server_protocol.ACK_ERROR)
        self.assertEqual(self.server_protocol.rank, -1)
        self.assertEqual(self.server_protocol.step, -1)

    @patch("msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.server.hashlib.md5")
    def test_post_process_success(self, _):
        self.shared_queue.maxsize = 1
        self.server_protocol.send_ack = MagicMock()

        def mock_send_ack_method1():
            self.server_protocol.consumer_queue.put(1)

        def mock_send_ack_method2():
            pass

        self.server_protocol.send_ack.side_effect = [mock_send_ack_method1, mock_send_ack_method2]
        self.server_protocol.check_sum = False
        self.server_protocol.obj_body = self.server_protocol.ACK_SUCCESS
        self.server_protocol.post_process()
        self.server_protocol.send_ack.assert_any_call(self.server_protocol.ACK_SUCCESS)

    def test_handle_with_stop(self):
        self.server_protocol.send_ack = MagicMock()
        self.server_protocol.handle_with_stop()
        self.server_protocol.send_ack.assert_called_once_with(self.server_protocol.ACK_STOP_CONFIRM)
        self.assertEqual(self.server_protocol.consumer_queue.get(), self.server_protocol.ACK_KILL_PROCESS)

    def test_reset_env(self):
        self.server_protocol.obj_length = 10
        self.server_protocol.sequence_number = 1
        self.server_protocol.rank = 2
        self.server_protocol.step = 3
        self.server_protocol.reset_env()
        self.assertEqual(self.server_protocol.obj_length, None)
        self.assertEqual(self.server_protocol.sequence_number, -1)
        self.assertEqual(self.server_protocol.rank, -1)
        self.assertEqual(self.server_protocol.step, -1)

    def test_dataReceived(self):
        self.server_protocol.buffer = io.BytesIO()
        self.server_protocol.post_process = MagicMock()
        unpack_mode = '!Q'
        header = struct.pack(unpack_mode, 10)
        header += struct.pack(unpack_mode, 1)
        header += struct.pack(unpack_mode, 2)
        header += struct.pack(unpack_mode, 3)

        self.server_protocol.dataReceived(header)

        self.assertEqual(self.server_protocol.obj_length, 10)
        self.assertEqual(self.server_protocol.sequence_number, 1)
        self.assertEqual(self.server_protocol.rank, 2)
        self.assertEqual(self.server_protocol.step, 3)


class TestMessageServerFactory(unittest.TestCase):
    def setUp(self):
        self.message_server_factory = MessageServerFactory()

    def test_is_all_connection_closed(self):
        all_conn_closed = self.message_server_factory.is_all_connection_closed()
        self.assertTrue(all_conn_closed)

        self.message_server_factory.transport_dict = {"test1": 1}
        all_conn_closed = self.message_server_factory.is_all_connection_closed()
        self.assertFalse(all_conn_closed)
