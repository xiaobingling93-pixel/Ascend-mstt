import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase
from common.logger import logger


class ArgMaxWithValue(Cell):
    def __init__(self, axis=0, keep_dims=False):
        super().__init__()
        self.argmaxwithvalue = P.ArgMaxWithValue(axis, keep_dims)
    
    def construct(self, input_x):
        return self.argmaxwithvalue(input_x)
    

class AddUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        len_args = len(args)
        self.axis = self.kwargs.get("axis") if self.kwargs else 0
        self.keep_dims = self.kwargs.get("keep_dims") if self.kwargs else False
    def forward_mindspore_impl(self, *args):
        x = args[0]
        net = ArgMaxWithValue(self.axis, self.keep_dims)
        out = net(x)
        return out
    
    def forward_pytorch_impl(self, *args):
        input_pt_x = args[0]
        value, index = torch.max(input_pt_x, self.axis, self.keep_dims)
        return (index, value)