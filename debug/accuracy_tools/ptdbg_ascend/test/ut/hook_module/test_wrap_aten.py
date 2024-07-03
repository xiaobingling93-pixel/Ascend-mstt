import unittest
import torch
from ptdbg_ascend.hook_module.wrap_aten import AtenOPTemplate, AtenOPPacketTemplate


def noop_hook_wrapper(name):
    def noop_hook(module, in_feat, out_feat):
        pass

    return noop_hook


if torch.__version__.split("+")[0] > '2.0':
    class TestWrapAten(unittest.TestCase):
        def setUp(self):
            self.aten_op = AtenOPPacketTemplate(torch.ops.aten.convolution, noop_hook_wrapper)

        def test_atenop_attribute(self):
            self.assertEqual(self.aten_op.default.op, torch.ops.aten.convolution.default)
            self.assertEqual(self.aten_op.out.op, torch.ops.aten.convolution.out)

        def test_atenop_forward(self):
            image = torch.randn(4, 3, 24, 24)
            kernel = torch.randn(10, 3, 3, 3)
            functional_out = torch.nn.functional.conv2d(image, kernel, stride=[1, 1],
                                                        padding=[1, 1], dilation=[1, 1], groups=1, bias=None)
            aten_out = self.aten_op(image, kernel, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            self.assertTrue(torch.all(functional_out == aten_out))

        def test_atenop_overload_forward(self):
            image = torch.randn(4, 3, 24, 24)
            kernel = torch.randn(10, 3, 3, 3)
            functional_out = torch.nn.functional.conv2d(image, kernel, stride=[1, 1],
                                                        padding=[1, 1], dilation=[1, 1], groups=1, bias=None)
            aten_out = self.aten_op.default(image, kernel, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
            self.assertTrue(torch.all(functional_out == aten_out))

        def test_atenop_nonattr(self):
            self.assertRaises(AttributeError, getattr, self.aten_op, "foo")

        def test_atenop_overloads(self):
            self.assertEqual(self.aten_op.overloads(), self.aten_op.opPacket.overloads())
