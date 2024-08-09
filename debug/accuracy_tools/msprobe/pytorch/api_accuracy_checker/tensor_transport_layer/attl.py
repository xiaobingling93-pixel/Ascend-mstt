import io
import os.path
import time
import re
from pathlib import Path
from multiprocessing import Queue
from typing import Optional, Union, Dict, Any
from collections import namedtuple
from dataclasses import dataclass

import torch

from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.client import TCPClient
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.server import TCPServer
from msprobe.pytorch.common.utils import logger
from msprobe.pytorch.common.utils import save_pt
from msprobe.core.common.utils import remove_path


ApiData = namedtuple('ApiData', ['name', 'args', 'kwargs', 'result', 'step', 'rank'],
                     defaults=['unknown', None, None, None, 0, 0])
BufferType = Union[ApiData, Dict[str, Any], str]  # Union[Tensor, Tuple[Optional[Tensor]]]


@dataclass
class ATTLConfig:
    is_benchmark_device: bool
    connect_ip: str
    connect_port: int
    # storage_config
    nfs_path: str = None
    tls_path: str = None
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
        self.check_attl_config()
        if self.session_config.nfs_path:
            self.nfs_path = Path(self.session_config.nfs_path)
        elif self.session_config.is_benchmark_device:

            self.socket_manager = TCPServer(self.session_config.connect_port,
                                            self.data_queue,
                                            self.session_config.check_sum,
                                            self.session_config.tls_path)
            self.socket_manager.start()
        elif need_dump:
            self.socket_manager = TCPClient(self.session_config.connect_ip,
                                            self.session_config.connect_port,
                                            self.session_config.check_sum,
                                            self.session_config.tls_path)
            self.socket_manager.start()

    def check_attl_config(self):
        if self.session_config.nfs_path:
            if os.path.exists(self.session_config.nfs_path):
                return
            else:
                raise Exception(f"nfs path {self.session_config.nfs_path} doesn't exists.")
        ipv4_pattern = "([1-9]?\d|1\d{2}|2[0-4]\d|25[0-5])(\.([1-9]?\d|1\d{2}|2[0-4]\d|25[0-5])){3}$"
        if not re.match(ipv4_pattern, self.session_config.connect_ip):
            raise Exception(f"host {self.session_config.connect_ip} is invalid.")
        if not (0 < self.session_config.connect_port <= 65535):
            raise Exception(f"port {self.session_config.connect_port} is invalid.")

    def stop_serve(self):
        if isinstance(self.socket_manager, TCPServer):
            self.socket_manager.stop()

    def send(self, buffer: BufferType) -> None:
        """
        npu major in 'send' (client)
        """
        # know receiver receive and go next
        if isinstance(buffer, ApiData):
            buffer = move2target_device(buffer, torch.device('cpu'))

        if 'device' in buffer.kwargs:
            buffer.kwargs.pop('device')
        rank = buffer.rank if hasattr(buffer, "rank") and buffer.rank is not None else 0
        step = buffer.step if hasattr(buffer, "step") else 0
        io_buff = io.BytesIO()
        try:
            torch.save(buffer, io_buff)
        except Exception as e:
            logger.info(f"{buffer.name} can not be saved, skip: {e}")
            return
        data = io_buff.getvalue()
        self.socket_manager.add_to_sending_queue(data, rank=rank, step=step)

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
                self.logger.warning("there is something error. please check it. %s", e)
            if isinstance(buffer, bytes):
                return None
            if isinstance(buffer, str):
                return buffer

        return buffer

    def upload(self, buffer: BufferType):
        if isinstance(buffer, ApiData):
            buffer = move2target_device(buffer, torch.device('cpu'))
            file_path = os.path.join(self.session_config.nfs_path, buffer.name + ".pt")
        else:
            file_path = os.path.join(self.session_config.nfs_path, buffer + f"_{int(time.time())}")

        try:
            save_pt(buffer, file_path)
        except Exception as e:
            self.logger.warning("there is something error in save_pt. please check it. %s", e)

    def download(self):
        for file_type in ("start*", "*.pt", "end*"):
            cur_file = next(self.nfs_path.glob(file_type), None)
            if cur_file is not None:
                break

        if cur_file is None:
            return None
        else:
            buffer = None
            try:
                buffer = torch.load(cur_file)
            except Exception as e:
                self.logger.warning("there is something error. please check it. %s", e)
            remove_path(cur_file)
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
    elif "return_types" in str(type(obj)):
        return move2device_exec(tuple(obj), device)
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
    new_results = move2device_exec(buffer.result, target_device)

    if target_device == torch.device('cpu') or target_device == "cpu":
        return ApiData(buffer.name, tuple(new_args), new_kwargs, new_results, buffer.step, buffer.rank)
    else:
        return ApiData(buffer.name, tuple(new_args), new_kwargs, buffer.result, buffer.step, buffer.rank)
