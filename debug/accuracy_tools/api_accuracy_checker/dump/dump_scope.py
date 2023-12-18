# dump范围控制
import torch 
from torch.utils.data.dataloader import _BaseDataLoaderIter 
from api_accuracy_checker.dump.dump import DumpUtil 
from api_accuracy_checker.common.config import msCheckerConfig


def iter_tracer(func):
    def func_wrapper(*args, **kwargs):
        DumpUtil.dump_switch = "OFF"
        result = func(*args, **kwargs)
        DumpUtil.incr_iter_num_maybe_exit()
        DumpUtil.call_num += 1
        return result 
    return func_wrapper

if msCheckerConfig.enable_dataloader:
    _BaseDataLoaderIter.__next__ = iter_tracer(torch.utils.data.dataloader._BaseDataLoaderIter.__next__)