import struct
import hashlib
import time
import io
from threading import Thread
from twisted.internet import reactor, protocol, endpoints
from api_accuracy_checker.common.utils import logger, print_info_log


class TCPServer:
    def __init__(self, port, shared_queue, check_sum=False) -> None:
        self.port = port
        self.shared_queue = shared_queue
        self.check_sum = check_sum
        self.factory = MessageServerFactory()
        self.reactor_thread = None

    def start(self):
        self.factory.protocol = self.build_protocol
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

    @staticmethod
    def run_reactor():
        reactor.run(installSignalHandlers=False)

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
        print_info_log(f"已连接客户端{self.transport.getPeer()}")

    def connectionLost(self, reason):
        self.factory.transport_dict.pop(self.transport, None)
        if len(self.factory.transport_dict) == 0:
            self.consumer_queue.put(b'KILL_')

        print_info_log(f"REASON: {reason} 与客户端{self.transport.getPeer()} 断开连接, "
                       f"self.factory.transport_dict: {len(self.factory.transport_dict)}")

    def send_ack(self, ack_info):
        self.transport.write(ack_info)

    def post_process(self):
        send_busy_ack = False
        while self.consumer_queue.full():
            if not send_busy_ack:
                self.send_ack(self.ACK_BUSY +
                              self.sequence_number.to_bytes(8, byteorder='big') +
                              self.rank.to_bytes(8, byteorder='big') +
                              self.step.to_bytes(8, byteorder='big'))
                logger.debug("sending BUSY ACK")
            send_busy_ack = True
            time.sleep(0.1)

        obj_key = str(self.sequence_number) + "_" + str(self.rank) + "_" + str(self.step)

        if self.check_sum:
            recv_md5 = hashlib.md5(self.obj_body).hexdigest()
            if recv_md5 == self.obj_md5:
                if self.obj_body == self.ACK_STOP:
                    self.handle_with_stop()
                else:
                    self.send_ack(self.ACK_SUCCESS +
                                  self.sequence_number.to_bytes(8, byteorder='big') +
                                  self.rank.to_bytes(8, byteorder='big') +
                                  self.step.to_bytes(8, byteorder='big'))
                if obj_key in self.sequence_number_dict:
                    logger.debug(f"这是一次异常的重传，可以忽略。 {obj_key}, {self.sequence_number_dict}")
                else:
                    self.sequence_number_dict[obj_key] = self.obj_md5
                    self.consumer_queue.put(self.obj_body, block=True)
            else:
                logger.debug(
                    f"Error: 接收数据有问题，流水号{self.sequence_number}  : expected {self.obj_md5}, but get {recv_md5}")

                self.send_ack(self.ACK_ERROR + self.sequence_number.to_bytes(8, byteorder='big') +
                              self.rank.to_bytes(8, byteorder='big') +
                              self.step.to_bytes(8, byteorder='big'))
        else:
            if self.obj_body == self.ACK_STOP:
                self.handle_with_stop()
            else:
                self.send_ack(self.ACK_SUCCESS + self.sequence_number.to_bytes(8, byteorder='big') +
                              self.rank.to_bytes(8, byteorder='big') +
                              self.step.to_bytes(8, byteorder='big'))
            if obj_key in self.sequence_number_dict:
                logger.debug("这是一次异常的重传，可以忽略。 {obj_key}, {self.sequence_number_dict}")
            else:
                self.sequence_number_dict[obj_key] = self.obj_md5
                self.consumer_queue.put(self.obj_body, block=True)

        self.reset_env()
        finish_time = time.time()
        logger.debug(f"finish_time: {finish_time - self.start_time}")

    def handle_with_stop(self):
        logger.debug(f"接收到停止传输信号 TCP{self.transport.getPeer()}")
        self.send_ack(self.ACK_STOP_CONFIRM +
                      self.sequence_number.to_bytes(8, byteorder='big') +
                      self.rank.to_bytes(8, byteorder='big') +
                      self.step.to_bytes(8, byteorder='big'))
        if len(self.factory.transport_dict) == 0:
            _rank, _step, _sequence_number = 0, 0, 100000000
            ack_kill = self.ACK_KILL_PROCESS + \
                       _sequence_number.to_bytes(8, byteorder='big') + \
                       _rank.to_bytes(8, byteorder='big') + \
                       _step.to_bytes(8, byteorder='big')
            for trans in self.factory.transport_list:
                trans.write(ack_kill)
            logger.debug(f"发送KILL信息给{self.transport.getPeer()}")
            self.consumer_queue.put(b'KILL_')
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
        while True:
            if self.obj_length is None and len(self.buffer.getvalue()) >= self.length_width * 4:
                # 解析长度信息
                self.start_time = time.time()
                self.obj_length = struct.unpack('!Q', self.buffer.read(self.length_width))[0]
                self.sequence_number = struct.unpack('!Q', self.buffer.read(self.length_width))[0]
                self.rank = struct.unpack('!Q', self.buffer.read(self.length_width))[0]
                self.step = struct.unpack('!Q', self.buffer.read(self.length_width))[0]
                self.tell += self.length_width * 4
                logger.debug(
                    f"流水号: {self.sequence_number}; RANK: {self.rank}; STEP: {self.step}; Length: {self.obj_length}")

            check_sum_and_md5 = self.check_sum and self.obj_length is not None and self.obj_md5 is None and len(
                self.buffer.getvalue()) - self.tell >= self.md5_width
            if check_sum_and_md5:
                # 提取数据包
                self.obj_md5 = self.buffer.read(self.md5_width).decode()
                self.tell += self.md5_width
                logger.debug(f"MD5: {self.obj_md5}")

            current_length = len(self.buffer.getvalue()) - self.tell

            if self.obj_length is not None and 0 < self.obj_length <= current_length:
                self.obj_body = self.buffer.read(self.obj_length)

                self.tell += self.obj_length
                self.buffer = io.BytesIO(self.buffer.getvalue()[self.tell:])
                self.buffer.seek(0)
                self.tell = 0
                recv_data_time = time.time()
                logger.debug(f"self.sequence_number {self.sequence_number} "
                             f"recv_data_time {recv_data_time - self.start_time}")

                if self.obj_body == self.ACK_STOP:
                    _transport = self.factory.transport_dict.pop(self.transport, None)
                    logger.debug(f"接收到b'STOP_' self.sequence_number {self.sequence_number} ")
                self.post_process()
                break
            else:
                break


class MessageServerFactory(protocol.ServerFactory):
    def __init__(self) -> None:
        self.transport_dict = {}
        self.transport_list = []

    def is_all_connection_closed(self):
        return len(self.transport_dict) == 0
