
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

import os
from collections import defaultdict
from functools import wraps

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from msprobe.pytorch.api_accuracy_checker.common.utils import ApiData
from msprobe.pytorch.common.utils import get_tensor_rank
from msprobe.core.common.const import Const
from msprobe.pytorch.common.log import logger
from msprobe.core.common.file_utils import load_yaml


def singleton(cls):
    _instance = {}

    @wraps(cls)
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner


@singleton
class Counter:
    def __init__(self) -> None:
        self.index_dict = defaultdict(int)


counter = Counter()
yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "torch_ops_config.yaml")
yaml_file = load_yaml(yaml_path)


class AccuracyCheckerDispatch(TorchDispatchMode):
    def __init__(self, attl):
        super(AccuracyCheckerDispatch, self).__init__()
        self.attl = attl
        self.counter = counter
        self.aten_ops_blacklist = []
        self.npu_adjust_autogard = []
        self.aten_ops_blacklist = yaml_file.get('aten_ops_blacklist', [])
        self.npu_adjust_autogard = yaml_file.get('npu_adjust_autogard', [])

    def __torch_dispatch__(self, func, types, args=None, kwargs=None):
        func_name_split_list = func.__name__.split(Const.SEP)
        aten_api = func_name_split_list[0]
        self.enable_autogard(aten_api)
        if aten_api in self.aten_ops_blacklist:
            npu_out = func(*args, **kwargs)
            return npu_out

        res = func(*args, **kwargs)
        cur_rank = get_tensor_rank(args, res)
        cur_api_number = self.counter.index_dict[aten_api]
        api_name = f'{Const.ATEN}{Const.SEP}{aten_api}{Const.SEP}{cur_api_number}'
        logger.info(f"tools is dumping api: {api_name}, rank: {cur_rank}")
        api_data = ApiData(api_name, args, kwargs, res, 0, cur_rank)
        if "device" in api_data.kwargs:
            api_data.kwargs.pop("device")
        if self.attl.nfs_path:
            self.attl.upload(api_data)
        else:
            self.attl.send(api_data)
        self.counter.index_dict[aten_api] += 1

        return res
    
    def enable_autogard(self, aten_api):
        if aten_api in self.npu_adjust_autogard:
            torch._C._dispatch_tls_set_dispatch_key_excluded(torch._C.DispatchKey.AutogradFunctionality, False)


def dispatch4data(func, attl, status):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not status:
            return func(*args, **kwargs)
        with AccuracyCheckerDispatch(attl):
            res = func(*args, **kwargs)
            return res

    return wrapper


def run_ut_dispatch(attl, status, is_recompute=False):
    """
    This function called by online_run_ut.
    It is used to enable or disable dispatch for torch.autograd.backward function.

    Args:
        attl (ATTL):  online_run_ut class ATTL, which is used to upload or send api data to server.
        status (bool): True means enable dispatch, False means disable dispatch.
        is_recompute (bool): Flag of recompute, which is conflicted with aten api, then skip dispatch4data.
    """
    if is_recompute:
        return
    torch.autograd.backward = dispatch4data(torch.autograd.backward, attl, status)
