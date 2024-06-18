import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase


class Concat(Cell):
    def __init__(self, axis=0):
        super().__init__()
        self.concat = P.Concat(axis)

    def construct(self, input_x):
        return self.concat(input_x)
    
class ConcatUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        self.axis = self.kwargs.get("axis") if self.kwargs else 0

    def forward_mindspore_impl(self, *args):
        x = args[0]
        net = Concat(self.axis)
        out = net(x)
        return out
    
    def forward_pytorch_impl(self, *args):
        input_pt_x = args[0]
        output = torch.cat(input_pt_x, self.axis)
        return output