import unittest
import torch
from atat.pytorch.hook_module.wrap_aten import AtenOPTemplate, AtenOPPacketTemplate


def hook(name):
    def forward_pre_hook(nope, input, kwargs):
            return input, kwargs
    def forward_hook():
            return 2
    def backward_hook():
            pass
    
    return forward_pre_hook, forward_hook, backward_hook


if torch.__version__.split("+")[0] > '2.0':
    class TestWrapAten(unittest.TestCase):
        def setUp(self):
            self.aten_op = AtenOPPacketTemplate(torch.ops.aten.convolution, hook)
        
        def test_atenop_attribute(self):
            self.setUp()
            self.assertEqual(self.aten_op.default.op, torch.ops.aten.convolution.default)
            self.assertEqual(self.aten_op.out.op, torch.ops.aten.convolution.out)

        def test_atenop_forward(self):
            self.setUp()
            image = torch.randn(4, 3, 24, 24)
            kernel = torch.randn(10, 3, 3, 3)
            functional_out = torch.nn.functional.conv2d(image, kernel, stride=[1, 1],
                                                        padding=[1, 1], dilation=[1, 1], groups=1, bias=None)
            aten_out = self.aten_op(image, kernel, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            self.assertTrue(aten_out == functional_out)

        def test_atenop_overload_forward(self):
            self.setUp()
            image = torch.randn(4, 3, 24, 24)
            kernel = torch.randn(10, 3, 3, 3)
            functional_out = torch.nn.functional.conv2d(image, kernel, stride=[1, 1],
                                                        padding=[1, 1], dilation=[1, 1], groups=1, bias=None)
            aten_out = self.aten_op(image, kernel, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            self.assertTrue(aten_out == functional_out)

        def test_atenop_nonattr(self):
            self.setUp()
            self.assertRaises(AttributeError, getattr, self.aten_op, "foo")

        def test_atenop_overloads(self):
            self.setUp()
            self.assertEqual(self.aten_op.overloads(), self.aten_op.opPacket.overloads())



            