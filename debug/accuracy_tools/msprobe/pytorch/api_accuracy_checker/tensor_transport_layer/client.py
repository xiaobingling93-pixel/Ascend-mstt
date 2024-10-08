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

import hashlib
import io
import struct
import time
import os
import signal
from queue import Queue
from threading import Thread
from typing import Union

from twisted.internet import reactor, protocol, endpoints
from twisted.protocols.basic import FileSender

from msprobe.pytorch.common.utils import logger
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.utils import struct_unpack_mode as unpack_mode, \
    str_to_bytes_order as bytes_order

MAX_SENDING_QUEUE_SIZE = 20


class TCPDataItem:
    def __init__(self, data,
                 sequence_number: int,
                 rank: int = 0,
                 step: int = 0):
        self.raw_data = data
        self.sequence_number = sequence_number
        self.rank = rank
        self.step = step
        self.retry_times = 0
        self.pending_time = 0
        self.busy_time = 0


class TCPClient:
    ACK_SUCCESS = b"OK___"
    ACK_ERROR = b"ERROR"
    ACK_BUSY = b"BUSY_"
    ACK_STOP = b"STOP_"
    ACK_STOP_CONFIRM = b"OVER_"
    ACK_KILL_PROCESS = b"KILL_"

    QUEUE_PENDING_TIME = 60
    RESEND_RETRY_TIMES = 2  # 最大重传数
    RESEND_TIMER_TIME = 5  # 接收ACK超时定时器
    RESEND_PENDING_TIME = 60  # 连续pending时间超过1分钟则放弃该数据

    def __init__(self, host="localhost", port=8000, check_sum=False, tls_path=None):
        self.send_queue = Queue(MAX_SENDING_QUEUE_SIZE)
        self.resend_dict = dict()
        self.host = host
        self.port = port
        self.tls_path = tls_path
        self.factory = None
        self.sequence_number = 0
        self.signal_exit = False
        self.tcp_manager = ClientProtocol(ack_queue_size=100,
                                          chunk_size=655360,
                                          check_sum=check_sum,
                                          tls=self.tls_path)
        self.send_thread = Thread(target=self._sending_queue_data)
        self.send_thread.setDaemon(True)
        self.send_thread.start()
        self.destroy_thread = Thread(target=self._destroy_queue_data)
        self.destroy_thread.setDaemon(True)
        self.destroy_thread.start()

    @staticmethod
    def run_reactor():
        reactor.run(installSignalHandlers=False)

    def check_tls_path(self):
        client_key = os.path.join(self.tls_path, "client.key")
        client_crt = os.path.join(self.tls_path, "client.crt")
        if not os.path.exists(client_key):
            raise Exception(f"client_key: {client_key} is not exists.")
        if not os.path.exists(client_crt):
            raise Exception(f"client_crt: {client_crt} is not exists.")
        return client_key, client_crt

    def start(self):
        def conn_callback(cur_protocol):
            if cur_protocol.transport and cur_protocol.transport.getPeer().host == self.host:
                logger.debug(f"Process: {os.getpid()} connects to server successfully.")
            else:
                logger.warning(f"Process: {os.getpid()} fails to connect to server. ")
                raise ConnectionError(f"Failed to connect to {self.host}.")

        def conn_err_callback(failure):
            self.signal_exit = True
            time.sleep(1)
            reactor.stop()
            logger.error(f"Failed to connected {self.host} {self.port}. Reason is {failure.getErrorMessage()}")

        def cur_protocol():
            return self.tcp_manager

        self.factory = MessageClientFactory()
        self.factory.protocol = cur_protocol
        if self.tls_path:
            from twisted.internet import ssl
            client_key, client_crt = self.check_tls_path()
            client_context_factory = ssl.DefaultOpenSSLContextFactory(client_key, client_crt)
            endpoint = endpoints.SSL4ClientEndpoint(reactor, self.host, self.port, client_context_factory)
        else:
            endpoint = endpoints.TCP4ClientEndpoint(reactor, self.host, self.port)
        d = endpoint.connect(self.factory)
        d.addCallback(conn_callback)
        d.addErrback(conn_err_callback)

        reactor_thread = Thread(target=self.run_reactor, daemon=True)
        reactor_thread.start()

    def send_after_queue_empty(self, data):
        while not self._ready_to_exit():
            if not self.tls_path:
                self.add_to_sending_queue(data)
            else:
                for _ in range(MAX_SENDING_QUEUE_SIZE):
                    self.add_to_sending_queue(data)
            time.sleep(2)

    def check_client_alive(self):
        return self.factory.num_connections > 0

    def stop(self):
        self.tcp_manager.connection_timeout()

    def send_stop_signal(self):
        self.send_after_queue_empty(self.ACK_STOP)
        while not self._ready_to_exit():
            if not self.check_client_alive():
                break
            time.sleep(1)

    def add_to_sending_queue(self, data: Union[bytes, TCPDataItem], rank: int = 0, step: int = 0):
        if self._ready_to_exit():
            return

        send_data = data
        if not isinstance(data, TCPDataItem):
            send_data = TCPDataItem(data=data,
                                    sequence_number=self.sequence_number,
                                    rank=rank,
                                    step=step)
            self.sequence_number += 1
        try:
            self.send_queue.put(send_data, block=True, timeout=self.QUEUE_PENDING_TIME)
        except Exception as e:
            logger.error(f"send_queue put send_data timeout, rank: {send_data.rank}, step: {send_data.step},"
                         f"sequence_number: {send_data.sequence_number}, send_queue size: {self.send_queue.qsize()},"
                         f"{str(e)}")

    def _send_data(self, data: TCPDataItem):
        self.tcp_manager.send_wrapped_data(data.raw_data,
                                           sequence_number=data.sequence_number,
                                           rank=data.rank,
                                           step=data.step
                                           )

    def _sending_queue_data(self):
        while True:
            if not self.tcp_manager.is_connected:
                continue

            while self.send_queue.qsize() > 0:
                if self._ready_to_exit():
                    break
                if len(self.resend_dict) < MAX_SENDING_QUEUE_SIZE:
                    data_obj = self.send_queue.get()
                    resend_key = str(data_obj.sequence_number) + "_" + str(data_obj.rank) + "_" + str(data_obj.step)
                    logger.debug(f"get {resend_key} from send_queue, and send to server.")
                    self._send_data(data_obj)
                    if resend_key not in self.resend_dict.keys():
                        # Send data for the first time
                        self.resend_dict[resend_key] = data_obj
                else:
                    time.sleep(0.1)

            if self._ready_to_exit():
                logger.debug("Successfully close sending process.")
                break
            time.sleep(0.1)

    def _destroy_queue_data(self):
        while True:
            if self._ready_to_exit():
                break

            while len(self.resend_dict) > 0 and self.tcp_manager.ack_queue.qsize() > 0:
                ack_info, seq_number, rank, step = self.tcp_manager.ack_queue.get()
                obj_key = str(seq_number) + "_" + str(rank) + "_" + str(step)
                current_item = self.resend_dict.get(obj_key)

                if current_item is None:
                    continue

                if ack_info == self.ACK_SUCCESS:
                    self.resend_dict.pop(obj_key)
                elif ack_info == self.ACK_BUSY:
                    logger.debug("RECV BUSY ACK")
                    if current_item.busy_time > 5:
                        self._resend_data(current_item)
                    else:
                        current_item.busy_time += 1
                elif ack_info == self.ACK_ERROR:
                    logger.debug("RECV ERROR ACK")
                    self._resend_data(current_item)
                elif ack_info == self.ACK_STOP_CONFIRM:
                    logger.debug("RECV STOP ACK")
                    self.factory.num_connections -= 1

                break

            time.sleep(0.1)

    def _resend_data(self, data: TCPDataItem):
        if data.retry_times < self.RESEND_RETRY_TIMES:
            data.retry_times += 1
            logger.debug(f"Resend data seq number: {data.sequence_number}")
            self.add_to_sending_queue(data)
        else:
            self.resend_dict.pop(data.sequence_number)
            logger.debug(f"SKIP send sequence number {data.sequence_number} after retry {data.retry_times} times!")

    def _pending_data(self, data: TCPDataItem):
        if data.pending_time >= self.RESEND_PENDING_TIME:
            self.resend_dict.pop(data.sequence_number)
            logger.debug(f"SKIP send sequence number {data.sequence_number} after pending {data.pending_time} times!")
            return

        # wait time is 100MB per second
        pending_time = max(1, len(data.raw_data) // (2 ** 20 * 50))
        data.pending_time += pending_time
        time.sleep(pending_time)

    def _ready_to_exit(self):
        return self.signal_exit or self.tcp_manager.signal_exit


class ClientProtocol(protocol.Protocol):
    TIMEOUT = 60 * 10

    def __init__(self, ack_queue_size=100, chunk_size=65536, check_sum=False, tls=None):
        self.buffer = io.BytesIO()
        self.is_connected = False
        self.check_sum = check_sum
        self.tell = 0
        self.ack_queue = Queue(maxsize=ack_queue_size)
        self.file_sender = FileSender()
        self.file_sender.CHUNK_SIZE = chunk_size
        self.signal_exit = False
        self.defer = None
        self.kill_process = False
        self.ack = None

        self.timeout_call = None

        self.tls = tls
        self.send_buffer = b""
        self.buffer_cnt = 0

    def dataReceived(self, data):
        if self.timeout_call.active():
            self.timeout_call.reset(self.TIMEOUT)

        self.buffer.seek(0, 2)
        self.buffer.write(data)
        self.buffer.seek(self.tell)
        while True:
            if len(self.buffer.getvalue()) >= 29:  # 5 + 8 * 3
                ack = self.buffer.read(5)
                self.ack = ack
                seq_number = struct.unpack(unpack_mode, self.buffer.read(8))[0]
                rank = struct.unpack(unpack_mode, self.buffer.read(8))[0]
                step = struct.unpack(unpack_mode, self.buffer.read(8))[0]
                logger.debug(f"receive 流水号: {seq_number}; RANK: {rank}; STEP: {step}; ACK: {ack}")
                if ack == b"KILL_":
                    self.kill_process = True
                    logger.debug(f"接收到KILL信号, PID {os.getpid()}")
                if ack == b"OVER_":
                    self.factory.num_connections -= 1
                self.tell += 29
                if not self.ack_queue.full():
                    self.ack_queue.put((ack, seq_number, rank, step))
                    self.buffer = io.BytesIO(self.buffer.getvalue()[self.tell:])
                    self.tell = 0
                else:
                    time.sleep(0.1)
            else:
                break

    def send_wrapped_data(self, data, sequence_number: int = 0, rank: int = 0, step: int = 0):
        length = len(data)
        md5_hash = hashlib.md5(data).hexdigest() if self.check_sum else ""
        data_meaasge = length.to_bytes(8, byteorder=bytes_order) + \
                       sequence_number.to_bytes(8, byteorder=bytes_order) + \
                       rank.to_bytes(8, byteorder=bytes_order) + \
                       step.to_bytes(8, byteorder=bytes_order) + \
                       md5_hash.encode() + \
                       data
        logger.debug(f"send 流水号: {sequence_number}; RANK: {rank}; STEP: {step}; LENGTH: {length}")

        while True:
            if self.defer is None or self.defer.called:
                self.defer = self.send_large_data(data_meaasge)
                break
            time.sleep(0.01)

    def send_large_data(self, data):

        if self.tls:
            self.send_buffer += data
            self.buffer_cnt += 1
            if self.buffer_cnt >= MAX_SENDING_QUEUE_SIZE:
                d = self.file_sender.beginFileTransfer(io.BytesIO(self.send_buffer), self.transport)
                self.send_buffer = b""
                self.buffer_cnt = 0
            else:
                d = None
        else:
            d = self.file_sender.beginFileTransfer(io.BytesIO(data), self.transport)
        return d

    def connection_timeout(self):
        if self.factory.num_connections <= 0:
            return

        self.factory.num_connections -= 1
        logger.debug(f"超时退出{self.transport.addr}, PID {os.getpid()}")
        self.transport.loseConnection()

    def connectionMade(self):
        self.timeout_call = reactor.callLater(self.TIMEOUT, self.connection_timeout)
        self.is_connected = True
        self.factory.num_connections += 1
        logger.info("successfully connect server")

    def connectionLost(self, reason):
        self.signal_exit = True
        self.factory.num_connections -= 1
        logger.info(f"Lost connection with server, reason is : {reason}")


class MessageClientFactory(protocol.ClientFactory):
    def __init__(self):
        self.num_connections = 0

    def clientConnectionFailed(self, connector, reason):
        logger.info(f"Fail to connection with server: {reason.getErrorMessage()}")
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        logger.info(f"Client lost connection with server: {reason.getErrorMessage()}")
        reactor.stop()
