import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase


class Equal(Cell):
    def __init__(self):
        super().__init__()
        self.equal = P.Equal()

    def construct(self, input_x, input_y):
        return self.equal(input_x, input_y)

class EqualUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)

    def forward_mindspore_impl(self, *args):
        x = args[0]
        y = args[1]
        net = Equal()
        out = net(x, y)
        return out

    def forward_pytorch_impl(self, *args):
        input_pt_x = args[0]
        input_pt_y = args[1]
        if not isinstance(input_pt_x, torch.Tensor):
            input_pt_x = torch.tensor(input_pt_x)
        out = torch.eq(input_pt_x, input_pt_y)
        return out