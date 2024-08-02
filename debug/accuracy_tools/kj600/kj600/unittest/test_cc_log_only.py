import os
import sys
sys.path.append(".")
import json
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from kj600.module_hook import TrainerMon
from kj600.unittest.cc_utils import *
from msprobe.core.common.file_check import FileOpen


with FileOpen(os.path.join(os.path.dirname(__file__), 'expected_cc_log.json'), 'r') as f:
    EXPECTED = json.load(f)

def test_all_gather(context, rank, world_size, async_op):
    a = torch.tensor([rank+1, rank+1], dtype=torch.float32, device=f'{DEVICE}:{rank}')
    data = [torch.empty_like(a) for _ in range(world_size)]
    dist.all_gather(data, a, group=dist.group.WORLD, async_op=async_op)
    assert_context(context.data, {}, rank)

def test_all_reduce(context, rank, world_size, async_op):
    a = torch.tensor([rank+1, rank+1], dtype=torch.float32, device=f'{DEVICE}:{rank}')
    dist.all_reduce(a, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=async_op)
    assert_context(context.data, {}, rank)


def main(rank, world_size):
    ddp_setup(rank, world_size)
    steps = 3
    async_op = False

    net = Model()
    monitor = TrainerMon("kj600/unittest/config_cc_logonly.json")
    monitor.hook_optimizer() # to enable tb
    optimizer = torch.optim.Adam(net.parameters())
    cc_context = monitor.cc_context
    try:
        for step in range(steps):
            print('step: ', step)
            test_all_gather(cc_context['all_gather'], rank, world_size, async_op)
            test_all_reduce(cc_context['all_reduce'], rank, world_size, async_op)
            optimizer.step()
    except Exception as e:
        assert step == 1
        assert e.__str__() == "exit after first step when print cc stack", e
        for k in EXPECTED.keys():
            assert [';'.join(stack) for stack in EXPECTED[k]] == list(monitor.cc_logged_stack[k])
        
  
if __name__ == '__main__':
    world_size=2
    torch.manual_seed(1234)
    mp.spawn(main, args=(world_size,), nprocs=world_size)

    