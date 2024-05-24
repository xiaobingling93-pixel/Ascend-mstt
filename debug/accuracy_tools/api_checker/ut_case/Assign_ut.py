import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase
from common.logger import logger


class Assign(Cell):
    def __init__(self):
        super().__init__()
        self.assign = P.Assign()
    
    def construct(self, variable, value):
        return self.assign(variable, value)
    

class AddUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
    
    def forward_mindspore_impl(self, *args):
        variable = args[0]
        value = args[1]
        net = Assign()
        out = net(variable, value)
        return out
    
    def forward_pytorch_impl(self, *args):
        variable = args[0]
        value = args[1]
        out = variable.copy(value)
        return out