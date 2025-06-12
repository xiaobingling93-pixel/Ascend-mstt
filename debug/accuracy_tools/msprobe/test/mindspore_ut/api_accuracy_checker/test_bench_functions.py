import unittest
import torch
import numpy as np

from msprobe.mindspore.api_accuracy_checker.bench_functions.flash_attention_score import (
    softmax_forward, softmax_grad, broadcast_kv, calculate_qk,
    fusion_attention_forward, fusion_attention_backward,
    parse_bsnd_args, convert_from_bnsd, convert_to_bnsd,
    generate_attn_mask, generate_kv, rebuid_softmax_by_qkv,
    rebuild_softmax_by_max_sum, FlashAttentionScore,
    npu_fusion_attention_forward_patch, npu_fusion_attention_backward_patch,
    FaForwardParams, FaBackwardParams, RebuildSoftmaxParams, GTYPE
)

class TestBenchFunctions(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_softmax_forward(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]], dtype=torch.float64)
        res, x_max, x_sum = softmax_forward(x)
        expected = torch.softmax(x, dim=-1)
        self.assertTrue(torch.allclose(res, expected, atol=1e-6))
        self.assertTrue(torch.allclose(x_max, x.max(dim=-1, keepdim=True)[0], atol=1e-6))
        manual_sum = torch.exp(x - x_max).sum(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(x_sum, manual_sum, atol=1e-6))

    def test_softmax_grad(self):
        x = torch.randn(4, 5, dtype=torch.float64)
        y, _, _ = softmax_forward(x)
        dp = torch.randn_like(y)
        grad = softmax_grad(dp, y)
        sum_grad = grad.sum(dim=-1)
        self.assertTrue(torch.allclose(sum_grad, torch.zeros_like(sum_grad), atol=1e-6))

    def test_broadcast_kv(self):
        B, N_kv, S, D = 1, 2, 3, 4
        num_heads = 4
        kv = torch.arange(B*N_kv*S*D, dtype=torch.float32).reshape(B, N_kv, S, D)
        out = broadcast_kv(num_heads, N_kv, kv, kv.dtype)
        self.assertEqual(out.shape, (B, num_heads, S, D))
        self.assertTrue(torch.equal(out[:, :2, :, :], kv[:, 0:1, :, :].expand(B,2,S,D)))
        self.assertTrue(torch.equal(out[:, 2:, :, :], kv[:, 1:2, :, :].expand(B,2,S,D)))

    def test_broadcast_kv_invalid(self):
        kv = torch.randn(1,3,4)  # 3D tensor
        with self.assertRaises(ValueError):
            broadcast_kv(4, 2, kv, kv.dtype)
        kv4d = torch.randn(1,2,3,4)
        with self.assertRaises(ValueError):
            broadcast_kv(0, 2, kv4d, kv4d.dtype)
        with self.assertRaises(ValueError):
            broadcast_kv(4, 3, kv4d, kv4d.dtype)
        with self.assertRaises(ValueError):
            broadcast_kv(3, 2, kv4d, kv4d.dtype)

    def test_calculate_qk_basic(self):
        q = torch.randn(2,2,3,4)
        k = torch.randn(2,2,3,4)
        scalar = 2.0
        qk = calculate_qk(q, k, None, None, scalar)
        expected = torch.matmul(q, k.permute(0,1,3,2)) * scalar
        self.assertTrue(torch.allclose(qk, expected, atol=1e-6))

    def test_calculate_qk_errors(self):
        q = torch.randn(2,2,3,4)
        # head_dim mismatch
        k = torch.randn(2,2,3,5)
        with self.assertRaises(ValueError):
            calculate_qk(q, k, None, None, 1.0)
        # too few dims
        q3 = torch.randn(2,3,4)
        k3 = torch.randn(2,3,4)
        with self.assertRaises(ValueError):
            calculate_qk(q3, k3, None, None, 1.0)

    def test_calculate_qk_with_pse_and_mask(self):
        q = torch.ones(1,1,2,2)
        k = torch.ones(1,1,2,2)
        pse = torch.ones(1,1,2,2)
        mask = torch.zeros(1,1,2,2)
        scalar = 1.0
        qk = calculate_qk(q, k, mask, pse, scalar)
        expected = (torch.matmul(q, k.permute(0,1,3,2)) + pse) * scalar + mask.bool() * (-40000.0)
        self.assertTrue(torch.allclose(qk, expected, atol=1e-6))

    def test_parse_bsnd_args_bsh(self):
        q = torch.randn(2,3,4)
        k = torch.randn(2,5,4)
        head_num = 2
        args = parse_bsnd_args(q, k, head_num, "BSH")
        b, s1, s2, n1, n2, d, h1, h2, dtype = args
        self.assertEqual((b, s1, s2, n1, n2, d, h1, h2), (2,3,5,2,2,2,4,4))

    def test_parse_bsnd_args_errors(self):
        q = torch.randn(2,3,0)  # leads to d=0
        k = torch.randn(2,5,0)
        with self.assertRaises(ValueError):
            parse_bsnd_args(q, k, 2, "BSH")
        with self.assertRaises(ValueError):
            parse_bsnd_args(q, k, 2, "TND")
        with self.assertRaises(ValueError):
            parse_bsnd_args(q, k, 0, "BSH")

    def test_convert_from_and_to_bnsd(self):
        B, N, S, D = 1, 2, 3, 4
        x = torch.arange(B*N*S*D).reshape(B, N, S, D)
        for layout in ["BSH","SBH","BSND","BNSD"]:
            out = convert_from_bnsd(x, layout)
            back = convert_to_bnsd(out, N, layout)
            self.assertTrue(torch.equal(back, x.to(GTYPE)))
        with self.assertRaises(ValueError):
            convert_to_bnsd(torch.randn(2,2), N, "TND")

    def test_generate_attn_mask_shapes_and_reverse(self):
        b,n1,s1,s2 = 1,1,3,3
        dtype = torch.float32
        for mode in range(5):
            mask = generate_attn_mask(mode, None, b, n1, s1, s2, 0, 0, dtype)
            self.assertEqual(mask.shape, (s1, s2))
        # reverse from full 2048 mask
        orig = torch.from_numpy(np.triu(np.ones([2048,2048]), k=1)).to(dtype)
        for mode in [2,3,4]:
            rev = generate_attn_mask(mode, orig, b, n1, s1, s2, 0, 0, dtype)
            self.assertEqual(rev.shape, (s1, s2))

    def test_generate_kv(self):
        key = torch.randn(1,2,3,4)
        value = torch.randn(1,2,3,4)
        k_new, v_new = generate_kv(key, value, 2, 2)
        self.assertTrue(torch.equal(k_new, key))
        k2, _ = generate_kv(key, value, 4, 2)
        self.assertEqual(k2.shape[1], 4)

    def test_rebuild_softmax_by_qkv(self):
        q = torch.randn(1,1,3,4)
        k = torch.randn(1,1,3,4)
        res = rebuid_softmax_by_qkv(q, k, None, None, 1.0)
        expected_shape = torch.matmul(q, k.permute(0,1,3,2)).shape
        self.assertEqual(res.shape, expected_shape)

    def test_rebuild_softmax_max_sum(self):
        q = torch.randn(1,1,3,4)
        k = torch.randn(1,1,3,4)
        softmax_max = torch.randn(1,1,1,2)
        softmax_sum = torch.randn(1,1,1,2)
        params = RebuildSoftmaxParams(q, k, None, None, 1.0, softmax_max, softmax_sum)
        res = rebuild_softmax_by_max_sum(params)
        repeat_dim = 4 // softmax_max.shape[-1]
        expected = torch.exp(torch.matmul(q, k.permute(0,1,3,2)) -
                              softmax_max.repeat(1,1,1,repeat_dim)).div(
            softmax_sum.repeat(1,1,1,repeat_dim)
        )
        self.assertTrue(torch.allclose(res, expected, atol=1e-6))
        # error when softmax_max length zero
        params_err = RebuildSoftmaxParams(q, k, None, None, 1.0, torch.randn(1,1,1,0), softmax_sum)
        with self.assertRaises(ValueError):
            rebuild_softmax_by_max_sum(params_err)

    def test_fusion_attention_forward_backward(self):
        B, N, S, D = 1, 2, 3, 4
        q = torch.randn(B, N, S, D)
        k = torch.randn(B, N, S, D)
        v = torch.randn(B, N, S, D)
        params = FaForwardParams(q, k, v, None, None, None, 1.0, 1.0)
        y, mx, sm = fusion_attention_forward(params)
        dq = torch.randn_like(y)
        bparams = FaBackwardParams(dq, q, k, v, torch.softmax(torch.randn_like(y), -1), None, None, 1.0, 1.0)
        dq_out, dk_out, dv_out = fusion_attention_backward(bparams)
        self.assertEqual(dq_out.shape, q.shape)
        self.assertEqual(dk_out.shape, k.shape)
        self.assertEqual(dv_out.shape, v.shape)
        # keep_prob zero errors
        params_bad = FaForwardParams(q, k, v, None, None, None, 1.0, 0)
        with self.assertRaises(ValueError):
            fusion_attention_forward(params_bad)
        bparams_bad = FaBackwardParams(dq, q, k, v, torch.softmax(torch.randn_like(y), -1), None, None, 1.0, 0)
        with self.assertRaises(ValueError):
            fusion_attention_backward(bparams_bad)

    def test_flash_attention_score_forward(self):
        fas = FlashAttentionScore()
        B, N, S, D = 1, 2, 3, 4
        q = torch.randn(B, S, N*D)
        k = torch.randn(B, S, N*D)
        v = torch.randn(B, S, N*D)
        out, max_vals, sum_vals = fas.forward(q, k, v, head_num=N, input_layout="BSH")
        self.assertIsInstance(out, torch.Tensor)
        self.assertIsInstance(max_vals, torch.Tensor)
        self.assertIsInstance(sum_vals, torch.Tensor)

    def test_npu_fusion_patches(self):
        # forward patch
        q = torch.randn(2,3,4)
        k = torch.randn(2,3,4)
        args, dims, newk = npu_fusion_attention_forward_patch(q, k, k, head_num=2, input_layout="BSH")
        self.assertIsInstance(args, list)
        with self.assertRaises(RuntimeError):
            npu_fusion_attention_forward_patch(q)
        # backward patch
        dx = torch.randn(2,3,4)
        args_b, dims_b, newk_b = npu_fusion_attention_backward_patch(q, k, k, dx, q, 2, input_layout="BSH")
        self.assertIsInstance(args_b, list)
        with self.assertRaises(ValueError):
            npu_fusion_attention_backward_patch(q, k)

if __name__ == '__main__':
    unittest.main()
