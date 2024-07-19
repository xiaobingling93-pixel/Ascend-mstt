import sys
sys.path.append(".")
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from kj600.module_hook import TrainerMon
from kj600.unittest.cc_utils import *

@wrap_reset
def test_all_gather(context, rank, target_rank, world_size, async_op):
    a = torch.tensor([rank+1, rank+1], dtype=torch.float32, device=f'{DEVICE}:{rank}')
    data = [torch.empty_like(a) for _ in range(world_size)]
    dist.all_gather(data, a, group=dist.group.WORLD, async_op=async_op)
    assert_context(context.data, {}, rank)

@wrap_reset
def test_all_reduce(context, rank, target_rank, world_size, async_op):
    a = torch.tensor([rank+1, rank+1], dtype=torch.float32, device=f'{DEVICE}:{rank}')
    dist.all_reduce(a, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=async_op)
    total = sum([i+1 for i in range(world_size)])
    sum_reduced = torch.tensor([total, total], dtype=torch.float32, device=f'{DEVICE}:{rank}')
    context.aggregate()
    if rank in target_rank:
        assert_context(context.data, {"post": [sum_reduced]}, rank)
    else:
        assert_context(context.data, {}, rank)

def main(rank, world_size):
    
    ddp_setup(rank, world_size)
    steps = 2
    async_op = False

    net = Model()
    monitor = TrainerMon("kj600/unittest/config_cc_codeline_ranks.json")
    target_rank = monitor.module_rank_list
    # monitor = None
    # monitor.hook_optimizer() # to enable tb
    optimizer = torch.optim.Adam(net.parameters())
    cc_context = monitor.cc_context
    for step in range(steps):
        print('setp: ', step)
        test_all_gather(cc_context['all_gather'], rank, target_rank, world_size, async_op)
        test_all_reduce(cc_context['all_reduce'], rank, target_rank, world_size, async_op)
        optimizer.step()
  
if __name__ == '__main__':
    world_size=2
    torch.manual_seed(1234)
    mp.spawn(main, args=(world_size,), nprocs=world_size)

    