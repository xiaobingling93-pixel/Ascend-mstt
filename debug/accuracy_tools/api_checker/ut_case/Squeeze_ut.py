import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase


class Squeeze(Cell):
    def __init__(self, axis=()):
        super().__init__()
        self.squeeze = P.Squeeze(axis)

    def construct(self, input_x):
        return self.squeeze(input_x)
    
class SqueezeUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        self.axis = self.kwargs.get("axis") if self.kwargs else ()

    def forward_mindspore_impl(self, *args):
        x = args[0]
        net = Squeeze(self.axis)
        out = net(x)
        return out
    
    def forward_pytorch_impl(self, *args):
        input_pt_x = args[0]
        if self.axis == ():
            output = torch.squeeze(input_pt_x)
        else:
            output = torch.squeeze(input_pt_x, self.axis)
        return output