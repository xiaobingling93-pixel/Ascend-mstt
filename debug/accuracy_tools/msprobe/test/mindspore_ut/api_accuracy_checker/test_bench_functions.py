import unittest
import torch
import numpy as np

from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker.bench_functions import (
    softmax_forward, softmax_grad, broadcast_kv, calculate_qk,
    fusion_attention_forward, fusion_attention_backward,
    parse_bsnd_args, convert_from_bnsd, convert_to_bnsd,
    generate_attn_mask, generate_kv, rebuid_softmax_by_qkv,
    rebuild_softmax_by_max_sum, FlashAttentionScore,
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
        # First half heads map to kv[:,0], second half to kv[:,1]
        self.assertTrue(torch.equal(out[:, :2, :, :], kv[:, 0:1, :, :].expand(B,2,S,D)))
        self.assertTrue(torch.equal(out[:, 2:, :, :], kv[:, 1:2, :, :].expand(B,2,S,D)))

    def test_calculate_qk_basic(self):
        q = torch.randn(2,2,3,4)
        k = torch.randn(2,2,3,4)
        scalar = 2.0
        qk = calculate_qk(q, k, None, None, scalar)
        expected = torch.matmul(q, k.permute(0,1,3,2)) * scalar
        self.assertTrue(torch.allclose(qk, expected, atol=1e-6))

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
        self.assertEqual(dtype, q.dtype)

    def test_convert_from_and_to_bnsd(self):
        B, N, S, D = 1, 2, 3, 4
        x = torch.arange(B*N*S*D).reshape(B, N, S, D)
        for layout in ["BSH","SBH","BSND","BNSD"]:
            out = convert_from_bnsd(x, layout)
            back = convert_to_bnsd(out, N, layout)
            self.assertTrue(torch.equal(back, x.to(GTYPE)))

    def test_generate_attn_mask_shapes(self):
        b,n1,s1,s2 = 1,1,3,3
        dtype = torch.float32
        for mode in range(5):
            mask = generate_attn_mask(mode, None, b, n1, s1, s2, 0, 0, dtype)
            self.assertEqual(mask.shape, (s1, s2))

    def test_generate_kv(self):
        key = torch.randn(1,2,3,4)
        value = torch.randn(1,2,3,4)
        k_new, v_new = generate_kv(key, value, 2, 2)
        self.assertTrue(torch.equal(k_new, key))
        self.assertTrue(torch.equal(v_new, value))
        k2, v2 = generate_kv(key, value, 4, 2)
        self.assertEqual(k2.shape[1], 4)
        self.assertEqual(v2.shape[1], 4)

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
        expected = torch.exp(torch.matmul(q, k.permute(0,1,3,2)) - softmax_max.repeat(1,1,1,repeat_dim)).div(
            softmax_sum.repeat(1,1,1,repeat_dim)
        )
        self.assertTrue(torch.allclose(res, expected, atol=1e-6))

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

if __name__ == '__main__':
    unittest.main()
