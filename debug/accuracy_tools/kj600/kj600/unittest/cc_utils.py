import os
from functools import partial
import torch
from torch import distributed as dist
from torch import nn
try:
    import torch_npu
    BACKEND = 'hccl'
    DEVICE = 'npu'
except:
    BACKEND = 'nccl'
    DEVICE = 'cuda'

from kj600.features import square_sum, get_max, get_min, get_zeros
from kj600.module_hook import CommunicationContext


OP_FUNCS = {
    "min": get_min,
    "max": get_max,
    "norm": square_sum,
    "zeros": partial(get_zeros, eps=1e-8)
}

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12346"
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)

def reset_context(context):
    if isinstance(context, CommunicationContext):
        context.reset()
    elif isinstance(context, dict):
        for op, v in context.items():
            v.reset()

def wrap_reset(func):
    def reset_and_test(*args, **kwargs):
        print(f"testing {func.__name__}")
        reset_context(args[0])
        res = func(*args, **kwargs)
        return res
    
    return reset_and_test

def assert_empty(data):
    assert len(data) == 0, f'data is not empty as expected'

def assert_nonempty(data):
    assert len(data) != 0, f'data is empty'

def assert_equal(a, b, rank, op_name=None, tag=None):
    if a.dim() == 0:
        assert a==b, f'inequal in rank {rank}: {a}, {b}, {op_name}, {tag}'
    else:
        assert torch.equal(a,b),  f'inequal in rank {rank}: {a},{b}'

def assert_inequal(a, b, rank):
    if a.dim() == 0:
        assert a!=b, f'equal in rank {rank}: {a},{b}'
    else:
        assert not torch.equal(a,b),  f'equal in rank {rank}: {a},{b}'

def assert_context(data, src, rank):
    if len(src) == 0:
        assert_empty(data)
    else:
        assert_nonempty(data)
    
    for op_name, tensors in data.items():
        for tag, tensor in tensors.items():
            prefix, idx = tag.split('_')
            idx = int(idx)
            assert_equal(tensor, OP_FUNCS[op_name](src[prefix][idx]), rank, op_name, tag)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Linear(2,2)

    def forward(self, x):
        return self.layer(x)