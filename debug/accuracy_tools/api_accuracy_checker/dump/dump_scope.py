# dump范围控制
import torch 
from torch.utils.data.dataloader import _BaseDataLoaderIter 
from api_accuracy_checker.dump.dump import DumpUtil 
from api_accuracy_checker.common.config import msCheckerConfig


def iter_tracer(original_next):
    def func_wrapper(*args, **kwargs):
        if msCheckerConfig.enable_dataloader:
            DumpUtil.dump_switch = "OFF"
            result = original_next(*args, **kwargs)
            DumpUtil.incr_iter_num_maybe_exit()
            DumpUtil.call_num += 1
            return result
        else:
            return original_next(*args, **kwargs)
    return func_wrapper

original_next_method = _BaseDataLoaderIter.__next__

_BaseDataLoaderIter.__next__ = iter_tracer(original_next_method)