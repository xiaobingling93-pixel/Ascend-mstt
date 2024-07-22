import sys
sys.path.append(".")
import time
import torch
from torch import nn
from torch import distributed as dist
import torch.multiprocessing as mp
from kj600.module_hook import TrainerMon
from kj600.unittest.cc_utils import *

DEBUG = False
DIM = 2
DTYPE = torch.float16

# 采集数据正确
# 通信结果正确

def test_broadcast(context, rank, async_op):
    a = torch.tensor([rank+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    local_a = a.clone()
    src = 0
    work = dist.broadcast(a, src, dist.group.WORLD, async_op)
    if work:
        work.wait()
    context.aggregate()
    if rank == src:
        assert_context(context.data, {'pre':[local_a], 'post':[a]}, rank)
        assert torch.equal(local_a, a), f"{local_a}, {a}"
    else:
        src_tensor = torch.tensor([src+1, src+1], dtype=DTYPE, device=f'{DEVICE}:{rank}')
        assert_context(context.data, {'pre': [local_a], 'post':[src_tensor]}, rank)
        assert_equal(src_tensor, a, rank)

@wrap_reset
def test_gather(context, rank, world_size, async_op):
    a = torch.tensor([rank+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    dst = 0
    if rank == dst:
        data = [torch.zeros_like(a) for _ in range(world_size)]
    else:
        data = None
    work = dist.gather(a, data, dst, group=dist.group.WORLD, async_op=async_op)
    if work:
        work.wait()
    context.aggregate()
    if rank == dst:
        assert_context(context.data, {'pre':[a, torch.zeros(world_size, 2, dtype=DTYPE)], 'post':[a, torch.stack(data)]}, rank)
        for i in range(world_size):
            local_a = torch.tensor([i+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
            assert_equal(data[i], local_a, rank)
    

@wrap_reset
def test_all_gather(context, rank, world_size, async_op):
    a = torch.tensor([rank+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    data = [torch.zeros_like(a, dtype=DTYPE) for _ in range(world_size)]
    work = dist.all_gather(data, a, group=dist.group.WORLD, async_op=async_op)
    if work:
        work.wait()
    context.aggregate()
    assert_context(context.data, {'pre':[torch.zeros(world_size, DIM, dtype=DTYPE), a], 'post':[torch.stack(data), a]}, rank)
    assert_equal(data[rank], a, rank)

@wrap_reset
def test_all_gather_into_tensor(context, rank, world_size, async_op):
    a = torch.tensor([rank+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    # concatenation
    data =  torch.zeros(world_size * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    res = torch.tensor([[i+1] for i in range(world_size)], dtype=DTYPE, device=f'{DEVICE}:{rank}').repeat(1, DIM)
    work = dist.all_gather_into_tensor(data, a, group=dist.group.WORLD, async_op=async_op)
    if work:
        work.wait()
    context.aggregate()
    assert_context(context.data, {'pre': [torch.zeros(world_size * DIM, dtype=DTYPE), a], 'post': [data, a]}, rank)
    assert_equal(data, res.flatten(), rank)

    context.reset()
    # concatenation
    data =  torch.zeros(world_size, DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    work = dist.all_gather_into_tensor(data, a, group=dist.group.WORLD, async_op=async_op)
    if work:
        work.wait()

    context.aggregate()
    assert_context(context.data, {'pre': [torch.zeros(world_size, DIM, dtype=DTYPE), a], 'post': [data, a]}, rank)
    assert_equal(data, res, rank)

@wrap_reset
def test_reduce(context, rank, world_size, async_op):
    a = torch.tensor([rank+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    local_a = a.clone()
    dst = 0
    work = dist.reduce(a, dst, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=async_op)
    if work:
        work.wait()
    context.aggregate()
    total = sum([i+1 for i in range(world_size)])
    res = torch.tensor([total] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    if rank == dst:
        assert_context(context.data, {'pre':[local_a], 'post':[res]}, rank)
        assert_equal(res, a, rank)
    else:
        assert_context(context.data, {'pre':[a], 'post':[a]}, rank)
        assert_equal(local_a, a, rank)

@wrap_reset
def test_all_reduce(context, rank, world_size, async_op):
    repeat = 2
    for _ in range(repeat): # test aggregate
        a = torch.tensor([rank+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
        local_a = a.clone()
        if rank == 0:
            time.sleep(6)
        work = dist.all_reduce(a, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=async_op)
        if work:
            work.wait()
    context.aggregate()
    total = sum([i+1 for i in range(world_size)])
    res = torch.tensor([total] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    assert_context(context.data, {'pre': [local_a.repeat(repeat)],'post': [res.repeat(repeat)]}, rank)
    assert_equal(res, a, rank)


@wrap_reset
def test_reduce_scatter(context, rank, world_size, async_op):
    a = torch.tensor([rank+1, rank+1], dtype=DTYPE, device=f'{DEVICE}:{rank}')
    output = torch.zeros_like(a)
    data = [a*(i+1) for i in range(world_size)]
    work = dist.reduce_scatter(output, data, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=async_op)
    if work:
        work.wait()
    context.aggregate()
    total = sum([i+1 for i in range(world_size)])
    res = (rank+1) * torch.tensor([total, total], dtype=DTYPE, device=f'{DEVICE}:{rank}')
    assert_context(context.data,{'pre': [torch.zeros_like(a), torch.stack(data)], 'post':[output, torch.stack(data)]}, rank)
    assert_equal(res, output, rank)


@wrap_reset
def test_reduce_scatter_tensor(context, rank, world_size, async_op):
    a = torch.tensor([rank+1] * DIM * world_size, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    output = torch.zeros(DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    work = dist.reduce_scatter_tensor(output, a, op=dist.ReduceOp.SUM, group=dist.group.WORLD, async_op=async_op)
    if work:
        work.wait()
    context.aggregate()
    total = sum([i+1 for i in range(world_size)])
    res = torch.tensor([total] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    assert_context(context.data,{'pre': [torch.zeros_like(a, dtype=DTYPE, device=f'{DEVICE}:{rank}'), a], 'post':[output, a]}, rank)
    assert_equal(res, output, rank)

@wrap_reset
def test_scatter(context, rank, world_size, async_op):
    a = torch.tensor([rank+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    local_a = a.clone()
    src = 0
    if rank == src:
        scatter_list = [10*torch.tensor([i+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}') for i in range(world_size)]
    else:
        scatter_list = None
    work = dist.scatter(a, scatter_list, src, group=dist.group.WORLD, async_op=async_op)
    if work:
        work.wait()
    context.aggregate()
    if rank == src:
        assert_context(context.data, {'pre': [local_a, torch.stack(scatter_list)], 'post': [a, torch.stack(scatter_list)]}, rank)
    else:
        assert_context(context.data, {'pre': [local_a], 'post': [a]}, rank)
    assert_equal(a, 10*torch.tensor([(rank+1)] * DIM ,dtype=DTYPE, device=f'{DEVICE}:{rank}'), rank)

## point2point
@wrap_reset
def test_send_recv(context, rank, world_size, async_op):
    """send from rank 0 to rank world_size-1"""
    if world_size<2:
        return 
    a = torch.tensor([rank+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    local_a = a.clone()
    src = 0
    dst = world_size-1
    if rank == src:
        dist.send(a, dst, group=dist.group.
                  WORLD)
        context['send'].aggregate()
        assert_context(context['send'].data, {'pre': [local_a], 'post': [a]}, rank)
        assert_equal(a, local_a, rank)
    if rank == dst:
        src_tensor = torch.tensor([src+1, src+1], dtype=DTYPE, device=f'{DEVICE}:{rank}')
        dist.recv(a, src, group=dist.group.
                  WORLD)
        context['recv'].aggregate()
        assert_context(context['recv'].data, {'pre':[local_a], 'post': [a]}, rank)
        assert_equal(a, src_tensor, rank)

@wrap_reset
def test_batch_isend_irecv(context, rank, world_size, async_op):
    send_tensor = torch.tensor([rank+1] * DIM, dtype=DTYPE, device=f'{DEVICE}:{rank}')
    recv_tensor = torch.zeros_like(send_tensor)
    send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1)%world_size)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size)%world_size)
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    context.aggregate()
    assert_context(context.data, {'pre': [torch.stack([send_tensor, torch.zeros_like(send_tensor)])], 'post':[torch.stack([send_tensor, recv_tensor])]}, rank)
    assert_equal( recv_tensor, torch.tensor([(rank - 1 + world_size)%world_size + 1] * DIM, device=f'{DEVICE}:{rank}'), rank)

def test_all(monitor, rank, world_size, async_op):
    cc_context = monitor.cc_context

    test_send_recv(cc_context, rank, world_size, async_op)
    test_broadcast(cc_context['broadcast'], rank, async_op)
    test_gather(cc_context['gather'], rank, world_size, async_op)
    test_all_gather(cc_context['all_gather'], rank, world_size, async_op)
    test_all_gather_into_tensor(cc_context['all_gather_into_tensor'], rank, world_size, async_op)
    test_reduce(cc_context['reduce'], rank, world_size, async_op)
    test_all_reduce(cc_context['all_reduce'], rank, world_size, async_op)
    test_reduce_scatter(cc_context['reduce_scatter'], rank, world_size, async_op)
    test_reduce_scatter_tensor(cc_context['reduce_scatter_tensor'], rank, world_size, async_op)
    test_scatter(cc_context['scatter'], rank, world_size, async_op)
    test_batch_isend_irecv(cc_context['batch_isend_irecv'], rank, world_size, async_op)
    
    
def main(rank, world_size):
    
    ddp_setup(rank, world_size)
    if rank == 0 and DEBUG:
        import debugpy
        debugpy.listen(5678)
        debugpy.wait_for_client()
    steps = 2

    net = Model()
    monitor = TrainerMon("kj600/unittest/config_cc.json", opt_ty="Megatron_Float16OptimizerWithFloat16Params")
    # monitor = None
    # monitor.hook_optimizer() # to enable tb
    optimizer = torch.optim.Adam(net.parameters())
    for step in range(steps):
        print('setp: ', step)
        test_all(monitor, rank, world_size, False)
        test_all(monitor, rank, world_size, True)
        optimizer.step()
        

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Linear(2,2)

    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    if len(sys.argv)>1:
        DEBUG = sys.argv[1]
    world_size=4
    torch.manual_seed(1234)
    mp.spawn(main, args=(world_size,), nprocs=world_size)

    