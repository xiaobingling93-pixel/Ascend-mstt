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

import os.path
import struct
import hashlib
import time
import io
from threading import Thread

from twisted.internet import reactor, protocol, endpoints

from msprobe.pytorch.common.utils import logger
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.utils import cipher_list, \
    struct_unpack_mode as unpack_mode, str_to_bytes_order as bytes_order


class TCPServer:
    def __init__(self, port, shared_queue, check_sum=False, tls_path=None) -> None:
        self.port = port
        self.shared_queue = shared_queue
        self.check_sum = check_sum
        self.tls_path = tls_path
        self.factory = MessageServerFactory()
        self.reactor_thread = None

    @staticmethod
    def run_reactor():
        reactor.run(installSignalHandlers=False)

    def check_tls_path(self):
        server_key = os.path.join(self.tls_path, "server.key")
        server_crt = os.path.join(self.tls_path, "server.crt")
        if not os.path.exists(server_key):
            raise Exception(f"server_key: {server_key} is not exists.")
        if not os.path.exists(server_crt):
            raise Exception(f"server_crt: {server_crt} is not exists.")
        return server_key, server_crt

    def start(self):
        self.factory.protocol = self.build_protocol

        if self.tls_path:
            from OpenSSL import SSL
            from twisted.internet import ssl
            server_key, server_crt = self.check_tls_path()
            server_context_factory = ssl.DefaultOpenSSLContextFactory(server_key, server_crt, SSL.TLSv1_2_METHOD)
            server_context_ = server_context_factory.getContext()
            server_context_.set_cipher_list(cipher_list)
            server_context_.set_options(SSL.OP_NO_RENEGOTIATION)
            endpoint = endpoints.SSL4ServerEndpoint(reactor, self.port, server_context_factory)
        else:
            endpoint = endpoints.TCP4ServerEndpoint(reactor, self.port)
        endpoint.listen(self.factory)
        self.reactor_thread = Thread(target=self.run_reactor, daemon=True)
        self.reactor_thread.start()

    def is_running(self):
        return not self.factory.is_all_connection_closed()

    def stop(self):
        self.factory.doStop()
        reactor.callFromThread(reactor.sigInt, 2)
        self.reactor_thread.join()

    def build_protocol(self):
        return ServerProtocol(self.shared_queue, self.check_sum)


class ServerProtocol(protocol.Protocol):
    ACK_SUCCESS = b"OK___"
    ACK_ERROR = b"ERROR"
    ACK_BUSY = b"BUSY_"
    ACK_STOP = b"STOP_"
    ACK_STOP_CONFIRM = b"OVER_"
    ACK_KILL_PROCESS = b"KILL_"

    def __init__(self, shared_queue, check_sum=False):
        self.start_time = None
        self.buffer = io.BytesIO()
        self.consumer_queue = shared_queue
        self.check_sum = check_sum
        self.length_width = 8
        self.md5_width = 32
        self.obj_length = None
        self.tell = 0
        self.obj_md5 = None
        self.obj_body = None
        self.sequence_number = -1
        self.rank = -1
        self.step = -1
        self.sequence_number_dict = dict()

    def connectionMade(self):
        self.buffer = io.BytesIO()
        self.obj_length = None
        self.tell = 0
        self.obj_md5 = None
        self.obj_body = None
        self.factory.transport_dict[self.transport] = 1
        self.factory.transport_list.append(self.transport)
        logger.info(f"Connected to {self.transport.getPeer()} successfully.")

    def connectionLost(self, reason):
        self.factory.transport_dict.pop(self.transport, None)
        if len(self.factory.transport_dict) == 0:
            self.consumer_queue.put(self.ACK_KILL_PROCESS)

        logger.info(f"Lost connection with {self.transport.getPeer()}. Reason is: {reason} 与客户端 断开连接, "
                    f"current connection number is: {len(self.factory.transport_dict)}")

    def send_ack(self, ack_info):
        ack_message = b"".join([
            ack_info,
            self.sequence_number.to_bytes(8, byteorder=bytes_order),
            self.rank.to_bytes(8, byteorder=bytes_order),
            self.step.to_bytes(8, byteorder=bytes_order)
        ])
        self.transport.write(ack_message)

    def post_process(self):
        send_busy_ack = False
        while self.consumer_queue.full():
            if not send_busy_ack:
                self.send_ack(self.ACK_BUSY)
                logger.debug("sending BUSY ACK")
            send_busy_ack = True
            time.sleep(0.1)

        obj_key = str(self.sequence_number) + "_" + str(self.rank) + "_" + str(self.step)

        recv_md5 = hashlib.md5(self.obj_body).hexdigest()
        if self.check_sum and recv_md5 != self.obj_md5:
            # when needs check md5 and check no pass, indicates received data error, send b"ERROR" to client.
            logger.debug(f"Error:接收数据有问题，流水号{self.sequence_number}, expected {self.obj_md5}, but get {recv_md5}")
            self.send_ack(self.ACK_ERROR)
        else:
            if self.obj_body == self.ACK_STOP:
                self.handle_with_stop()
            else:
                self.send_ack(self.ACK_SUCCESS)
            if obj_key in self.sequence_number_dict:
                logger.debug(f"这是一次异常的重传，可以忽略。 {obj_key}, {self.sequence_number_dict}")
            else:
                self.sequence_number_dict[obj_key] = self.obj_md5
                self.consumer_queue.put(self.obj_body, block=True)

        self.reset_env()
        finish_time = time.time()
        logger.debug(f"finish_time: {finish_time - self.start_time}")

    def handle_with_stop(self):
        logger.debug(f"接收到停止传输信号 TCP{self.transport.getPeer()}")
        self.send_ack(self.ACK_STOP_CONFIRM)
        if len(self.factory.transport_dict) == 0:
            _rank, _step, _sequence_number = 0, 0, 100000000
            ack_kill = self.ACK_KILL_PROCESS + \
                       _sequence_number.to_bytes(8, byteorder='big') + \
                       _rank.to_bytes(8, byteorder='big') + \
                       _step.to_bytes(8, byteorder='big')
            for trans in self.factory.transport_list:
                trans.write(ack_kill)
            logger.debug(f"发送KILL信息给{self.transport.getPeer()}")
            self.consumer_queue.put(self.ACK_KILL_PROCESS)
            time.sleep(2)

    def reset_env(self):
        self.obj_length = None
        self.sequence_number = -1
        self.rank = -1
        self.step = -1
        self.obj_md5 = None
        self.obj_body = None

    def dataReceived(self, data):
        self.buffer.seek(0, 2)
        self.buffer.write(data)
        self.buffer.seek(self.tell)

        # The first data packet is packet header, it contains obj_length, sequence_number, rank, step
        if self.obj_length is None and len(self.buffer.getvalue()) >= self.length_width * 4:
            self.start_time = time.time()
            self.obj_length = struct.unpack(unpack_mode, self.buffer.read(self.length_width))[0]
            self.sequence_number = struct.unpack(unpack_mode, self.buffer.read(self.length_width))[0]
            self.rank = struct.unpack(unpack_mode, self.buffer.read(self.length_width))[0]
            self.step = struct.unpack(unpack_mode, self.buffer.read(self.length_width))[0]
            self.tell += self.length_width * 4
            logger.debug(
                f"流水号: {self.sequence_number}; RANK: {self.rank}; STEP: {self.step}; Length: {self.obj_length}")

        # If needs check md5 but not parse md5 yet, read 32b md5 values
        check_sum_and_md5 = (self.check_sum
                             and self.obj_length is not None
                             and self.obj_md5 is None
                             and len(self.buffer.getvalue()) - self.tell >= self.md5_width)
        if check_sum_and_md5:
            self.obj_md5 = self.buffer.read(self.md5_width).decode()
            self.tell += self.md5_width
            logger.debug(f"MD5: {self.obj_md5}")

        current_length = len(self.buffer.getvalue()) - self.tell
        if self.obj_length is not None and 0 < self.obj_length <= current_length:
            # Current api data receive finished
            self.obj_body = self.buffer.read(self.obj_length)

            self.tell += self.obj_length
            self.buffer = io.BytesIO(self.buffer.getvalue()[self.tell:])
            self.buffer.seek(0)
            self.tell = 0
            recv_data_time = time.time()
            logger.debug(f"self.sequence_number {self.sequence_number} "
                         f"recv_data_time {recv_data_time - self.start_time}")

            if self.obj_body == self.ACK_STOP:
                # Indicates the current TCP link receives a STOP signal and remove from the transport_dict
                _transport = self.factory.transport_dict.pop(self.transport, None)
                logger.debug(f"接收到b'STOP_' self.sequence_number {self.sequence_number} ")
            self.post_process()


class MessageServerFactory(protocol.ServerFactory):
    def __init__(self) -> None:
        """
        transport_dict: links that have not completed data transmission.
        transport_list: Records all TCP links. Appends TCP link to the transport list
                        when a new TCP link is established.
        """
        self.transport_dict = {}
        self.transport_list = []

    def is_all_connection_closed(self):
        return len(self.transport_dict) == 0
