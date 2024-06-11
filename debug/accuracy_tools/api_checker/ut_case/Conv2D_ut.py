import numpy as np
import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
import torch.nn as nn
import torch.nn.functional as F
from ut_base import UTBase

class Conv2D(Cell):
    def __init__(self, out_channel, kernel_size, mode=1, pad_mode="valid", pad=0, stride=1, dilation=1, group=1, data_format="NCHW"):
        super(Conv2D, self).__init__()
        self.conv2d = P.Conv2D(out_channel, kernel_size, mode, pad_mode, pad, stride, dilation, group, data_format)
    
    def construct(self, input_x, weight):
        return self.conv2d(input_x, weight)

class Conv2DPytorch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2DPytorch, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=self._ensure_tuple(stride, kernel_size),
                              padding=0, dilation=self._ensure_tuple(dilation, kernel_size), groups=groups, bias=bias)
        self.padding_torch = self._to_padding(padding)
        self.flag = int(any(self.padding_torch))

    def forward(self, x):
        if self.flag:
            x = F.pad(x, self.padding_torch)
        return self.conv(x)
    
    def _ensure_tuple(self, value, ref_value):
        if isinstance(value, int):
            return (value,) * len(ref_value)
        if isinstance(value, (tuple, list)) and len(value) == len(ref_value):
            return tuple(value)
        elif isinstance(value, (tuple, list)) and len(value) == 4:
            return (value[2], value[3])
        raise ValueError(f"Invalid value for conversion to tuple: {value}")

    def _to_padding(self, padding):
        if isinstance(padding, int):
            return [padding] * 4
        if isinstance(padding, (tuple, list)) and len(padding) == 2:
            return [padding[0], padding[0], padding[1], padding[1]]
        if isinstance(padding, (tuple, list)) and len(padding) == 4:
            return list(padding)
        raise ValueError(f"Invalid padding value: {padding}")

class Conv2DUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        
        input_shape = self.args[0].shape
        weight_shape = self.args[1].shape
        
        self.out_channel = self.kwargs.get("out_channel", self.args[0])
        self.kernel_size = self.kwargs.get("kernel_size", self.args[1])
        self.mode = self.kwargs.get("mode", 1)
        self.pad_mode = self.kwargs.get("pad_mode", "valid")
        self.pad = self.kwargs.get("pad", 0)
        self.stride = self.kwargs.get("stride", 1)
        self.dilation = self.kwargs.get("dilation", 1)
        self.group = self.kwargs.get("group", 1)
        self.data_format = self.kwargs.get("format", "NCHW")
        
        if self.data_format == "NCHW":
            self.in_n, self.in_c, self.in_h, self.in_w = input_shape
            self.out_c, _, self.kernel_h, self.kernel_w = weight_shape
        else:  # NHWC
            self.in_n, self.in_h, self.in_w, self.in_c = input_shape
            self.out_c, self.kernel_h, self.kernel_w, _ = weight_shape

        self.stride = self._ensure_tuple(self.stride, (self.kernel_h, self.kernel_w))
        self.dilation = self._ensure_tuple(self.dilation, (self.kernel_h, self.kernel_w))

        if self.pad_mode == 'same':
            self.padding_torch = self._compute_same_padding(self.in_h, self.in_w, self.kernel_h, self.kernel_w, self.stride, self.dilation)
        else:
            self.padding_torch = [self.pad] * 4

    def _ensure_tuple(self, value, ref_value):
        if isinstance(value, int):
            return (value,) * len(ref_value)
        if isinstance(value, (tuple, list)) and len(value) == len(ref_value):
            return tuple(value)
        raise ValueError(f"Invalid value for conversion to tuple: {value}")

    def _compute_same_padding(self, h, w, kh, kw, stride, dilation):
        def compute_pad(dim, kernel, stride, dilation):
            pad = max((dim + stride - 1) // stride * stride - dim + (kernel - 1) * dilation, 0)
            return pad // 2, pad - pad // 2
        pad_h = compute_pad(h, kh, stride[0], dilation[0])
        pad_w = compute_pad(w, kw, stride[1], dilation[1])
        return [pad_w[0], pad_w[1], pad_h[0], pad_h[1]]

    def forward_mindspore_impl(self, *args):
        x = ms.Tensor(args[0])
        weight = ms.Tensor(args[1])
        net = Conv2D(self.out_channel, self.kernel_size, self.mode, self.pad_mode, self.pad, self.stride, self.dilation, self.group, self.data_format)
        out = net(x, weight)
        return out.asnumpy()
    
    def forward_pytorch_impl(self, *args):
        x, weight = args
        if self.data_format == 'NHWC':
            x = x.permute(0, 3, 1, 2)
            weight = weight.permute(0, 3, 1, 2)
        
        net = Conv2DPytorch(in_channels=self.in_c, out_channels=self.out_c,
                            kernel_size=(self.kernel_h, self.kernel_w),
                            stride=self.stride, padding=self.padding_torch,
                            dilation=self.dilation, groups=self.group, bias=False)
        
        net.conv.weight = nn.Parameter(weight)
        output = net(x)
        if self.data_format == 'NHWC':
            output = output.permute(0, 2, 3, 1)
        return output.detach()
