import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase


class Sub(Cell):
    def __init__(self):
        super().__init__()
        self.sub = P.Sub()

    def construct(self, input_x, input_y):
        return self.sub(input_x, input_y)
    
class SubUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)

    def forward_mindspore_impl(self, *args):
        x = args[0]
        y = args[1]
        net = Sub()
        out = net(x, y)
        return out
    
    def forward_pytorch_impl(self, *args):
        input_pt_x = args[0]
        input_pt_y = args[1]
        if isinstance(input_pt_x, bool):
            input_pt_x = int(input_pt_x)
        elif isinstance(input_pt_x, torch.Tensor) and input_pt_x.dtype == torch.bool:
            input_pt_x = input_pt_x.int()
        if isinstance(input_pt_y, bool):
            input_pt_y = int(input_pt_y)
        elif isinstance(input_pt_y, torch.Tensor) and input_pt_y.dtype == torch.bool:
            input_pt_y = input_pt_y.int()
        out = torch.sub(input_pt_x, input_pt_y)
        return out