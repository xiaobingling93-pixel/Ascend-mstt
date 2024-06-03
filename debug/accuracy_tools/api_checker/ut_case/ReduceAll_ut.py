import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase
from common.logger import logger


class ReduceAll(Cell):
    def __init__(self, axis, keep_dims=False):
        super().__init__()
        self.reduceall = P.ReduceAll(keep_dims=keep_dims)
        self.axis = axis

    def construct(self, x):
        return self.reduceall(x, self.axis)


class ReduceAllUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        self.keep_dims = self.kwargs.get("keep_dims") if self.kwargs else False

    def forward_mindspore_impl(self, *args):
        x = args[0]
        axis = args[1]
        net = ReduceAll(axis=axis, keep_dims=self.keep_dims)
        out = net(x)
        return out
    
    def forward_pytorch_impl(self, *args):
        x = args[0]
        axis = args[1]
        output = torch.all(x, dim=axis, keepdim=self.keep_dims)
        return output