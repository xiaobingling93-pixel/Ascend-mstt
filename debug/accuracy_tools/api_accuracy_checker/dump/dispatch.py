import os
from functools import wraps

import yaml
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from api_accuracy_checker.tensor_transport_layer.attl import ApiData
from api_accuracy_checker.common.utils import get_tensor_rank, print_info_log
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.dump.dump import DumpUtil
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen


def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner


@singleton
class Counter:
    def __init__(self) -> None:
        self.index_dict = {}


counter = Counter()
yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "torch_ops_config.yaml")
with FileOpen(yaml_path, 'r') as f:
    yaml_file = yaml.safe_load(f)


class AccuracyCheckerDispatch(TorchDispatchMode):
    def __init__(self):
        super(AccuracyCheckerDispatch, self).__init__()
        self.attl = DumpUtil.attl
        self.counter = counter
        self.aten_ops_blacklist = []
        self.npu_adjust_autogard = []
        self.aten_ops_blacklist = yaml_file.get('aten_ops_blacklist')
        self.npu_adjust_autogard = yaml_file.get('npu_adjust_autogard')

    def enable_autogard(self, aten_api):
        if aten_api in self.npu_adjust_autogard:
            torch._C._dispatch_tls_set_dispatch_key_excluded(torch._C.DispatchKey.AutogradFunctionality, False)

    def __torch_dispatch__(self, func, types, args=None, kwargs=None):
        func_name_split_list = func.__name__.split(".")
        aten_api = func_name_split_list[0]
        self.enable_autogard(aten_api)
        if aten_api in self.aten_ops_blacklist:
            npu_out = func(*args, **kwargs)
            return npu_out

        res = func(*args, **kwargs)
        cur_rank = get_tensor_rank(args, res)
        if cur_rank not in DumpUtil.rank_list:
            return res
        cur_api_number = self.counter.index_dict.setdefault(aten_api, 0)
        api_name = f'Aten*{aten_api}*{cur_api_number}'
        print_info_log(f"tools is dumping api: {api_name}")
        api_data = ApiData(api_name, args, kwargs, res, DumpUtil.call_num, cur_rank)
        if "device" in api_data.kwargs:
            api_data.kwargs.pop("device")
        if msCheckerConfig.nfs_path:
            self.attl.upload(api_data)
        else:
            self.attl.send(api_data)
        self.counter.index_dict[aten_api] += 1

        return res


def dispatch4data(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not DumpUtil.get_dump_switch() or DumpUtil.phase not in ("backward", "all") or \
                not msCheckerConfig.is_online:
            return func(*args, **kwargs)
        DumpUtil.set_dump_switch("OFF")
        with AccuracyCheckerDispatch():
            res = func(*args, **kwargs)
            DumpUtil.set_dump_switch("ON")
            return res

    return wrapper


torch.autograd.backward = dispatch4data(torch.autograd.backward)
