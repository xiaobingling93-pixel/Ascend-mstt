import unittest
import torch
from msprobe.pytorch.monitor.features import square_sum, get_min, get_mean, get_norm, get_max, get_zeros, \
    get_sign_matches, eff_rank, mNTK, lambda_max_subsample, cal_histc, get_nans


class TestMathFunctions(unittest.TestCase):
    def test_square_sum(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = square_sum(tensor)
        self.assertEqual(result, 14.0)

    def test_get_min(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = get_min(tensor)
        self.assertEqual(result, 1.0)

    def test_get_mean(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = get_mean(tensor)
        self.assertAlmostEqual(result, 2.0, places=1)

    def test_get_norm(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = get_norm(tensor)
        self.assertTrue(torch.allclose(result, torch.tensor(3.7417, dtype=torch.float64), atol=1e-4))

    def test_get_max(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = get_max(tensor)
        self.assertEqual(result, 3.0)

    def test_get_zeros(self):
        tensor = torch.tensor([1e-10, 2e-10, 3e-10])
        result = get_zeros(tensor, eps=1e-10)
        res = torch.allclose(result, torch.tensor(0.), atol=1e-1)
        self.assertTrue(res)

    def test_get_sign_matches(self):
        tensor_x = torch.tensor([1.0, -1.0, 1.0])
        tensor_y = torch.tensor([1.0, 1.0, -1.0])
        result = get_sign_matches(tensor_x, tensor_y)
        res = torch.allclose(result, torch.tensor(0.3333), atol=1e-4)
        self.assertTrue(res)

    def test_eff_rank(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
        result = eff_rank(tensor)
        res = torch.allclose(result, torch.tensor(2), atol=1e-1)
        self.assertTrue(res)

    def test_mNTK(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super(MockModule, self).__init__()

            def forward(self, x):
                return x + 1

        module = MockModule()
        tensor = torch.tensor([1.0])
        result = mNTK(module, tensor)
        res = torch.allclose(result, torch.tensor([[1.]]), atol=1e-1)
        self.assertTrue(res)

    def test_lambda_max_subsample(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super(MockModule, self).__init__()

            def forward(self, x):
                return x

        module = MockModule()
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = lambda_max_subsample(module, tensor)
        res = torch.allclose(result, torch.tensor(1.0), atol=1e-1)
        self.assertTrue(res)

    def test_cal_histc(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cal_histc(tensor, bins_total=3, min_val=1.0, max_val=5.0)
        self.assertEqual(result.size(), (3,))

    def test_get_nans(self):
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        result = get_nans(tensor)
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
