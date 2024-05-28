import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase


class BroadcastTo(Cell):
    def __init__(self, shape):
        super().__init__()
        self.broadcastto = P.BroadcastTo(shape)

    def construct(self, input_x):
        return self.broadcastto(input_x)
    
class BroadcastToUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        self.shape = self.kwargs.get("shape")
    
    def forward_mindspore_impl(self, *args):
        x = args[0]
        net = BroadcastTo(self.shape)
        out = net(x)
        return out

    def forward_pytorch_impl(self, *args):
        input_pt_x = args[0]
        output = torch.broadcast_to(input_pt_x, self.shape)
        return output