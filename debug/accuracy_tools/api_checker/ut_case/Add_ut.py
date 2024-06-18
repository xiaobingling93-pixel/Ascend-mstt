import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase
from common.logger import logger


class Add(Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()
    
    def construct(self, input_x, input_y):
        return self.add(input_x, input_y)
    

class AddUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
    
    def forward_mindspore_impl(self, *args):
        x = args[0]
        y = args[1]
        net = Add()
        out = net(x, y)
        return out
    
    def forward_pytorch_impl(self, *args):
        input_pt_x = args[0]
        input_pt_y = args[1]
        if not isinstance(input_pt_x, torch.Tensor):
            input_pt_x = torch.tensor(input_pt_x)
        output = torch.add(input_pt_x, input_pt_y)
        if output.dtype == torch.bfloat16:
            return output.float()
        return output