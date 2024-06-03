import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
from ut_base import UTBase


class OneHot(Cell):
    def __init__(self, axis=0):
        super.__init__()
        self.onehot = P.OneHot(axis)

    def construct(self, indices, depth, on_value, off_value):
        return self.onehot(indices, depth, on_value, off_value)
    
class OneHotUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        self.axis = self.kwargs.get("axis") if self.kwargs else -1

    def forward_mindspore_impl(self, *args):
        indices = args[0]
        depth = args[1]
        on_value = args[2]
        off_value = args[3]
        net = OneHot(self.axis)
        out = net(indices, depth, on_value, off_value)
        return out
    
    def forward_pytorch_impl(self, *args):
        indices = args[0]
        num_classes = args[1]
        on_value = args[2]
        off_value = args[3]
        dim = indices.ndim
        output = torch.nn.functional.one_hot(indices, num_classes)
        mask_for_ones = output == 1
        mask_for_zeros = output == 0
        output[mask_for_ones] = on_value
        output[mask_for_zeros] = off_value

        dims = tuple(range(dim))
        axis = self.axis
        if axis < 0:
            axis = axis + dim + 1
        dims = dims[:axis] + (dim,) + dims[axis:]
        output = output.permute(dims)
        return output

