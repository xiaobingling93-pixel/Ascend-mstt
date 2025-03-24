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

import glob
import os.path
import time
from multiprocessing import Queue
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

import torch

from msprobe.pytorch.api_accuracy_checker.common.utils import ApiData
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.client import TCPClient
from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.server import TCPServer
from msprobe.core.common.file_utils import remove_path
from msprobe.pytorch.common.utils import logger, save_api_data, load_api_data, save_pkl, load_pkl
from msprobe.core.common.decorator import recursion_depth_decorator

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
        self.nfs_path = None
        if self.session_config.nfs_path:
            self.nfs_path = self.session_config.nfs_path
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

    def stop_serve(self):
        if isinstance(self.socket_manager, TCPServer):
            self.socket_manager.stop()

    def send(self, buffer: BufferType) -> None:
        """
        npu major in 'send' (client)
        """

        # if tcp connection lost,
        if self.socket_manager.signal_exit:
            raise ConnectionError(f"Failed to connect to {self.session_config.connect_ip}.")

        # know receiver receive and go next
        if isinstance(buffer, ApiData):
            buffer = move2target_device(buffer, torch.device('cpu'))

        if 'device' in buffer.kwargs:
            buffer.kwargs.pop('device')
        rank = buffer.rank if hasattr(buffer, "rank") and buffer.rank is not None else 0
        step = buffer.step if hasattr(buffer, "step") else 0
        try:
            io_buff = save_api_data(buffer)
        except Exception as e:
            self.logger.info(f"{buffer.name} can not be saved, skip: {e}")
            return
        data = io_buff.getvalue()
        self.socket_manager.add_to_sending_queue(data, rank=rank, step=step)

    def recv(self, timeout_ms=0) -> Optional[BufferType]:
        buffer = ''
        while not buffer:
            if timeout_ms > 0:
                time.sleep(timeout_ms / 1000.0)
            if not buffer and not self.data_queue.empty():
                buffer = self.data_queue.get()
                break
            if not buffer and timeout_ms > 0:  # timeout is the only case we give up and return None
                break
            if self.message_end and self.data_queue.empty():
                buffer = b"KILL_CONFIRM"
                self.kill_progress = True
                break
            time.sleep(0.1)  # waiting outside the lock before next attempt
        if not buffer:
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
            try:
                buffer = load_api_data(buffer)
            except Exception as e:
                self.logger.warning("there is something error. please check it. %s", e)
            if isinstance(buffer, bytes):
                return ''
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
            save_pkl(buffer, file_path)
        except Exception as e:
            self.logger.warning("there is something error in save_pt. please check it. %s", e)

    def download(self):
        buffer = None
        cur_file = None
        for file_type in ("start*", "*.pt", "end*"):
            pattern = os.path.join(self.nfs_path, file_type)
            files = glob.glob(pattern)
            if len(files) > 0:
                cur_file = files[0]
                break

        if cur_file is not None:
            try:
                buffer = load_pkl(cur_file)
            except Exception as e:
                self.logger.warning("there is something error. please check it. %s", e)
            remove_path(cur_file)
        return buffer


@recursion_depth_decorator("move2device_exec")
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
