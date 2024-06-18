import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
import torch
from ut_base import UTBase


class MatMul(Cell):
    def __init__(self, transponse_a=False, transponse_b=False):
        super().__init__()
        self.matmul = P.MatMul(transponse_a, transponse_b)
    
    def construct(self, input_x, input_y):
        return self.matmul(input_x, input_y)
    

class MatMulUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        len_args = len(args)
        self.transpose_a = self.kwargs.get("transpose_a") if self.kwargs else False
        self.transpose_b = self.kwargs.get("transpose_b") if self.kwargs else False

    def forward_mindspore_impl(self, *args):
        x = args[0]
        y = args[1]
        net = MatMul(self.transpose_a, self.transpose_a)
        out = net(x, y)
        if out.dtype == mstype.bfloat16:
            return out.float().asnumpy()
        return out
    
    def forward_pytorch_impl(self, *args):
        input_pt_x = args[0]
        input_pt_y = args[1]
        if self.transpose_a:
            input_pt_x = torch.transpose(input_pt_x, 0, 1)
        if self.transpose_b:
            input_pt_y = torch.transpose(input_pt_y, 0, 1)
        output = torch.matmul(input_pt_x, input_pt_y)
        if output.dtype == torch.bfloat16:
            return output.float().numpy()
        return output