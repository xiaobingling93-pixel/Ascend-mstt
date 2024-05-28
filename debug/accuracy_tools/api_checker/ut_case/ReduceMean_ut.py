import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase
from common.logger import logger


class ReduceMean(Cell):
    def __init__(self, keep_dims=False):
        super().__init__()
        self.reducemean = P.ReduceMean(keep_dims)
    
    def construct(self, x, axis):
        return self.reducemean(x, axis)


class ReduceMeanUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        self.keep_dims = self.kwargs.get("keep_dims") if self.kwargs else False

    def forward_mindspore_impl(self, *args):
        x = args[0]
        axis = args[1]
        net = ReduceMean(self.keep_dims)
        out = net(x, axis)
        return out
    
    def forward_pytorch_impl(self, *args):
        x = args[0]
        axis = args[1]
        output = torch.mean(x, axis, self.keep_dims)
        return output