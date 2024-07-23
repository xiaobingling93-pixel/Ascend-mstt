import unittest
import torch
import torch.nn as nn
import torch_npu
from kj600.features import eff_rank


class TestFeatureCalculation(unittest.TestCase):
    def test_effective_rank(self):
        param = torch.randn(10, 10).npu()
        rank = eff_rank(param)
        self.assertTrue(rank.item() >= 1)

    def test_lambda_max(self):
        pass
        # input_dim = 10
        # hidden_dim = 100
        # output_dim = 1
        # num_samples = 100
        # X = torch.randn(num_samples, input_dim)
        # network = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim), 
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim)
        # )
        # Y = network(X)
        # Y.backward()
        # for name, param in network.named_parameters():
        #     lm = lambda_max(param)
        

if __name__ == "__main__":
    unittest.main()