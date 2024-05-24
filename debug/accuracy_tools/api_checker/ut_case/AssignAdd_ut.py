import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase
from common.logger import logger


class AssignAdd(Cell):
    def __init__(self):
        super().__init__()
        self.assignadd = P.AssignAdd()
    
    def construct(self, variable, value):
        return self.assignadd(variable, value)
    

class AssignAddUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)

    def forward_mindspore_impl(self, *args):
        x = args[0]
        y = args[1]
        net = AssignAdd()
        out = net(x, y)
        return out
    
    def forward_pytorch_impl(self, *args):
        variable = args[0]
        value = args[1]
        if not isinstance(value, torch.Tensor):
            value = torch.Tensor(value)
        out = torch.add(variable, value)
        return out