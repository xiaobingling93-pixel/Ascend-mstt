import io
import time
from multiprocessing import Queue
from typing import Optional, Union, Dict, Any
from collections import namedtuple
from dataclasses import dataclass

import torch

from api_accuracy_checker.tensor_transport_layer.client import TCPClient
from api_accuracy_checker.tensor_transport_layer.server import TCPServer
from api_accuracy_checker.common.utils import logger


ApiData = namedtuple('ApiData', ['name', 'args', 'kwargs', 'result', 'step', 'rank'],
                     defaults=['unknown', None, None, None, 0, 0])
BufferType = Union[ApiData, Dict[str, Any], str]  # Union[Tensor, Tuple[Optional[Tensor]]]


@dataclass
class ATTLConfig:
    # net_config: dict
    is_golden: bool
    connect_ip: str = "127.0.0.1"
    connect_port: int = 8006
    # storage_config
    check_sum: bool = True
    queue_size: int = 50


class ATTL:
    def __init__(self, session_id: str, session_config: ATTLConfig, need_dump=True) -> None:
        self.session_id = session_id
        self.session_config = session_config
        self.logger = logger
        self.socket_manager = None
        self.data_queue = Queue(maxsize=50)
        self.dequeue_list = []
        self.message_end = False
        self.kill_progress = False
        if self.session_config.is_golden:
            self.socket_manager = TCPServer(self.session_config.connect_port,
                                            self.data_queue,
                                            self.session_config.check_sum)
            self.socket_manager.start()
        elif need_dump:
            self.socket_manager = TCPClient(self.session_config.connect_ip,
                                            self.session_config.connect_port,
                                            self.session_config.check_sum)
            self.socket_manager.start()

    def stop_serve(self):
        if isinstance(self.socket_manager, TCPServer):
            self.socket_manager.stop()

    def client_handle(self, data, rank: int = 0, step: int = 0):
        self.socket_manager.add_to_sending_queue(data, rank=rank, step=step)

    def send(self, buffer: BufferType):
        """
        npu major in 'send' (client)
        """
        # know receiver receive and go next
        if isinstance(buffer, ApiData):
            buffer = move2target_device(buffer, torch.device('cpu'))
        
        if 'device' in buffer.kwargs:
            buffer.kwargs.pop('device')
        rank = buffer.rank if hasattr(buffer, "rank") else 0
        step = buffer.step if hasattr(buffer, "step") else 0
        io_buff = io.BytesIO()
        torch.save(buffer, io_buff)
        self.client_handle(io_buff.getvalue(), rank=rank, step=step)

    def recv(self, timeout_ms=0) -> Optional[BufferType]:
        buffer = None
        while buffer is None:
            if timeout_ms > 0:
                time.sleep(timeout_ms / 1000.0)
            if buffer is None and not self.data_queue.empty():
                buffer = self.data_queue.get()
                break
            if buffer is None and timeout_ms > 0:  # timeout is the only case we give up and return None
                break
            if self.message_end and self.data_queue.empty():
                buffer = b"KILL_CONFIRM"
                self.kill_progress = True
                break
            time.sleep(0.1)  # waiting outside the lock before next attempt
        if buffer is None:
            # this is a result of a timeout
            self.logger.info(f"RECEIVE API DATA TIMED OUT")
        else:
            if buffer == b"STOP_":
                return "STOP_"
            if buffer == b"KILL_":
                self.message_end = True
                return "STOP_"
            if buffer == b"KILL_CONFIRM":
                self.kill_progress = True
                return "KILL_"
            buffer = io.BytesIO(buffer)
            try:
                buffer = torch.load(buffer, map_location="cpu")
            except Exception as e:
                self.logger.error("there is something error. please check it. %s", e)
            if isinstance(buffer, bytes):
                return None
            if isinstance(buffer, str):
                return buffer

        return buffer


def move2device_exec(obj, device):
    if isinstance(obj, (tuple, list)):
        data_list = [move2device_exec(val, device) for val in obj]
        return data_list if isinstance(obj, list) else tuple(data_list)
    if isinstance(obj, dict):
        return {key: move2device_exec(val, device) for key, val in obj.items()}
    elif isinstance(obj, torch.Tensor):
        obj = obj.detach()
        if obj.device.type != device:
            obj = obj.to(device)
        return obj
    elif isinstance(obj, torch._C.device):
        return torch.device(device)
    else:
        return obj


def move2target_device(buffer: ApiData, target_device):
    # handle args
    new_args = move2device_exec(buffer.args, target_device)

    # handle kwargs
    new_kwargs = move2device_exec(buffer.kwargs, target_device)

    # handle result
    new_results = []
    res = buffer.result[0] if isinstance(buffer.result, (tuple, list)) else buffer.result
    if isinstance(res, torch.Tensor) and res.device.type != target_device:
        new_results.append(res.detach().to(target_device))
    else:
        new_results.append(res)

    if target_device == torch.device('cpu') or target_device == "cpu":
        return ApiData(buffer.name, tuple(new_args), new_kwargs, new_results[0], buffer.step, buffer.rank)
    else:
        return ApiData(buffer.name, tuple(new_args), new_kwargs, buffer.result, buffer.step, buffer.rank)
