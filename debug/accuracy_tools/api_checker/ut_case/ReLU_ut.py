import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
import torch.nn as nn
from ut_base import UTBase
from common.logger import logger


class ReLU(Cell):
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()
    
    def construct(self, input_x):
        return self.relu(input_x)
    

class ReLUUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
    
    def forward_mindspore_impl(self, *args):
        x = args[0]
        net = ReLU()
        out = net(x)
        return out
    
    def forward_pytorch_impl(self, *args):
        input_pt_x = args[0]
        net = nn.ReLU()
        output = net(input_pt_x)
        return output