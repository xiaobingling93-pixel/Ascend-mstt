import numpy as np
import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
import torch.nn as nn
import torch.nn.functional as F
from ut_base import UTBase
from common.utils import dtype_map


class Conv2D(Cell):
    def __init__(self, out_channel, kernel_size, mode=1, pad_mode="valid", pad=0, stride=1,
                 dilation=1, group=1, data_format="NCHW"):
        super.__init__()
        self.conv2d = P.Conv2D(out_channel, kernel_size, mode, pad_mode, pad, stride,
                               dilation, group, data_format)
    
    def construct(self, input_x, weight):
        return self.conv2d(input_x, weight)


class Conv2DPytoch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias):
        super(Conv2dPytorch, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=0,
                              dilation=dilation, groups=groups, bias=bias)
        self.padding_torch = padding
        self.flag = int(any(padding))

    def forward(self, x):
        if self.flag:
            x = F.pad(x, self.padding_torch)
        out = self.conv(x)
        return out
    

class Conv2DUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        
        dtype = self.args[0].dtype
        input_shape = self.args[0].shape
        weight_shape = self.args[1].shape
        self.dtype = dtype_map[dtype]
        
        self.out_channel = self.kwargs.get("out_channel") if self.kwargs else self.args[0]
        self.kernel_size = self.kwargs.get("kernel_size") if self.kwargs else self.args[1]
        self.mode = self.kwargs.get("mode") if self.kwargs else 1
        self.pad_mode = self.kwargs.get("pad_mode") if self.kwargs else "valid"
        self.pad = self.kwargs.get("pad") if self.kwargs else 0
        self.stride = self.kwargs.get("stride") if self.kwargs else 1
        self.dilation = self.kwargs.get("dilation") if self.kwargs else 1
        self.group = self.kwargs.get("group") if self.kwargs.get("group") else 1
        self.data_format = self.kwargs.get("data_format") if self.kwargs else "NCHW"
        
        if self.data_format == "NCHW":
            self.in_n, self.in_c, self.in_h, self.in_w = input_shape
            self.out_c, self.kernel_c, self.kernel_h, self.kernel_w = weight_shape
        
        if self.data_format == "NHWC":
            self.in_n, self.in_h, self.in_w, self.in_c = input_shape
            self.out_c, self.kernel_h, self.kernel_w, self.kernel_c = weight_shape
            
        if isinstance(self.kernel_size, list):
            self.kernel_size = tuple(self.kernel_size)
        
        if isinstance(self.stride, list):
            self.stride = tuple(self.stride)
        
        if isinstance(self.dilation, list):
            self.dilation = tuple(self.dilation)
            
        self.dilation_torch = self.dilation
        
        if isinstance(self.dilation, int):
            self.dilation_torch = (self.dilation, self.dilation)
        
        if isinstance(self.pad, list):
            self.padding_torch = [self.pad, self.pad, self.pad, self.pad]
        elif isinstance(self.pad, tuple) and len(self.pad) == 2:
            self.padding_torch = self.pad
            self.padding = (self.pad[0], self.pad[0], self.pad[1], self.pad[1])
        else:
            self.padding_torch = self.pad[-2:] + self.pad[:2]
            self.padding = self.pad
        
        tmp_stride = self.stride
        if not isinstance(self.stride, tuple):
            tmp_stride = (self.stride, self.stride)
        
        if self.pad_mode == 'same':
            if self.in_h % tmp_stride[0] == 0:
                pad_along_height = max(
                    self.dilation_torch[0] * (self.kernel_h - 1) + 1 - tmp_stride[0],
                    0
                )
            else:
                pad_along_height = max(
                    self.dilation_torch[0] * (self.kernel_h - 1) + 1 - (self.in_h % tmp_stride[0]),
                    0
                )
            if self.in_w % tmp_stride == 0:
                pad_along_width = max(
                    self.dilation_torch[1] * (self.kernel_w - 1) + 1 - tmp_stride[1],
                    0
                )
            else:
                pad_along_width = max(
                    self.dilation_torch[1] * (self.kernel_w - 1) + 1 - (self.in_w % tmp_stride[1]),
                    0
                )
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            self.padding_torch = [pad_left, pad_right, pad_top, pad_bottom]
            
    def forward_mindspore_impl(self, *args):
        x = ms.Tensor(args[0])
        weight = ms.Tensor(args[1])
        net = Conv2D(self.out_channel, self.kernel_size, self.mode, self.pad_mode, self.pad,
                     self.stride, self.dilation, self.group, self.data_format)
        out = net(x, weight)
        return out.asnumpy()
    
    def forward_pytorch_impl(self, *args):
        x = args[0]
        weight = args[1]
        if self.data_format == 'NCHW':
            x = x.astype(self.dtype)
            weight = weight.astype(self.dtype)
        
        if self.data_format == 'NHWC':
            x = x.transpose(0, 3, 1, 2).astype(self.dtype)
            weight = x.transpose(0, 3, 1, 2).astype(self.dtype)
        
        net = Conv2DPytoch(in_channels=self.in_c, out_channels=self.out_c,
                           kernel_size=(self.kernel_h, self.kernel_w),
                           stride=self.stride, padding=self.padding_torch,
                           dilation=self.dilation_torch, groups=self.group, bias=False)
        
        net.conv.register_parameter('weight', nn.Parameter(weight))
        output = net(x)
        if self.data_format == 'NHWC':
            output = output.transpose(0, 2, 3, 1)
        return output.detach().numpy().astype(self.dtype)

    