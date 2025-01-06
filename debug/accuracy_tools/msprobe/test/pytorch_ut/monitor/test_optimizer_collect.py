import unittest
from collections import defaultdict
from unittest.mock import Mock, patch, MagicMock

import torch
from msprobe.pytorch.monitor.optimizer_collect import OptimizerMon, \
    OptimizerMonFactory, DummyOptimizerMon, \
    MixPrecisionOptimizerMon, MegatronDistributedOptimizerMon, MegatronFP32OptimizerMon, \
    DeepSpeedZeroOptimizerStage0Mon, DeepSpeedZeroOptimizerStage1or2Mon, DeepSpeedZeroOptimizerStage3Mon

from msprobe.pytorch.monitor.utils import MVResult, MVGradResult


class TestOptimizerMon(unittest.TestCase):

    def setUp(self) -> None:
        # 初始化需要的monitor, torch_opt, params2name等对象
        self.monitor = Mock()
        self.monitor.mv_distribution = True
        self.monitor.mg_direction = True
        self.monitor.ur_distribution = True
        self.monitor.update_heatmap_visualizer = {'param1': Mock(), 'param2': Mock()}
        self.monitor.ratio_heatmap_visualizer = {'param1': Mock(), 'param2': Mock()}

    def test_fetch_mv(self):
        optimizer_mon = OptimizerMon()
        res = optimizer_mon.fetch_mv(None, None, None)
        self.assertEqual(res, None)

    def test_fetch_mv_in_adam(self):
        self.torch_opt = Mock()
        self.torch_opt.state = {
            'param1': {'exp_avg': torch.tensor(0.1), 'exp_avg_sq': torch.tensor(0.2), 'step': torch.tensor(10)},
            'param2': {'exp_avg': torch.tensor(0.3), 'exp_avg_sq': torch.tensor(0.4), 'step': torch.tensor(20)}
        }
        self.torch_opt.param_groups = [{'step': 10}]
        self.torch_opt.defaults = {'betas': (0.9, 0.999), 'eps': 1e-8}
        self.params2name = {'param1': 'param1', 'param2': 'param2'}

        self.optimizer_mon = OptimizerMon()
        result = self.optimizer_mon._fetch_mv_in_adam(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(result, MVResult)

    @patch('msprobe.pytorch.monitor.optimizer_collect.dist')
    def test_fetch_mv_grad_in_adam(self, mock_dist):
        self.optimizer_mon = OptimizerMon()
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = defaultdict(str)
        self.name2indices = defaultdict(tuple)
        self.fp32_partitioned_groups_flat = defaultdict(torch.Tensor)

        # Mocking the dist.get_rank() and dist.get_world_size()
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 1

        # Mocking the wrapped_optimizer
        self.optimizer_mon.wrapped_optimizer = MagicMock()
        self.optimizer_mon.wrapped_optimizer.state = defaultdict(dict)
        self.optimizer_mon.wrapped_optimizer.averaged_gradients = defaultdict(torch.Tensor)
        self.optimizer_mon.wrapped_optimizer.partition_size = defaultdict(int)
        self.optimizer_mon.wrapped_optimizer.flatten_dense_tensors_aligned = MagicMock()
        self.optimizer_mon.wrapped_optimizer.flatten = MagicMock()

        # Mocking the torch_opt.param_groups
        self.torch_opt.param_groups = [{'step': 1, 'betas': (0.9, 0.999)},
                                       {'step': 2, 'betas': (0.9, 0.999)},
                                       {'step': 3, 'betas': (0.9, 0.999)}]

        # Mocking the monitor.mv_distribution, monitor.mg_direction, monitor.ur_distribution
        self.monitor.mv_distribution = True
        self.monitor.mg_direction = True
        self.monitor.ur_distribution = True

        # Mocking the monitor.update_heatmap_visualizer and monitor.ratio_heatmap_visualizer
        self.monitor.update_heatmap_visualizer = defaultdict(MagicMock)
        self.monitor.ratio_heatmap_visualizer = defaultdict(MagicMock)

        result = self.optimizer_mon._fetch_mv_grad_in_adam(self.monitor, self.torch_opt, self.params2name,
                                                           self.name2indices, self.fp32_partitioned_groups_flat)
        self.assertIsInstance(result, MVGradResult)


class TestMixPrecisionOptimizerMon(unittest.TestCase):

    def test_fetch_mv_with_fp16_to_fp32_param_and_mix_prec_opt(self):
        # init monitor, torch_opt ...
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        self.mix_prec_opt = MagicMock()
        self.mix_prec_opt.float16_groups = [MagicMock()]
        self.mix_prec_opt.fp32_from_float16_groups = [MagicMock()]
        self.optimizer = MixPrecisionOptimizerMon()
        self.optimizer.wrapped_optimizer = self.mix_prec_opt
        self.optimizer.fp16_to_fp32_param = {}

        # Mock _fetch_mv_in_adam method and set a fixed return value
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv_in_adam = MagicMock(return_value=mv_result)
        self.optimizer._fetch_mv_in_adam = self.mock_fetch_mv_in_adam

        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.mock_fetch_mv_in_adam.assert_called_once_with(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)


class TestMegatronDistributedOptimizerMon(unittest.TestCase):
    def setUp(self):
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        self.mock_wrapped_optimizer = MagicMock()
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv_in_adam = MagicMock(return_value=mv_result)
        self.optimizer = MegatronDistributedOptimizerMon()

    def test_fetch_mv_with_valid_optimizer(self):
        self.mock_wrapped_optimizer.model_float16_groups = [MagicMock()]
        self.mock_wrapped_optimizer.shard_fp32_from_float16_groups = [MagicMock()]
        self.optimizer.wrapped_optimizer = self.mock_wrapped_optimizer
        self.optimizer._fetch_mv_in_adam = self.mock_fetch_mv_in_adam

        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)

    def test_fetch_mv_with_invalid_optimizer(self):
        self.optimizer.wrapped_optimizer = Mock()
        self.optimizer._fetch_mv_in_adam = self.mock_fetch_mv_in_adam

        with self.assertRaises(Exception):
            self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)


class TestCommonFetchMv(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        self.mock_wrapped_optimizer = MagicMock()

    def test_megatron_fp32_optimizer_mon(self):
        self.optimizer = MegatronFP32OptimizerMon()
        self.optimizer.wrapped_optimizer = self.mock_wrapped_optimizer
        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)

    def test_deepspeed_zero_optimizer_stage0_mon(self):
        self.optimizer = DeepSpeedZeroOptimizerStage0Mon()
        self.optimizer.wrapped_optimizer = self.mock_wrapped_optimizer
        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)

    def test_dummy_optimizer_mon(self):
        self.optimizer = DummyOptimizerMon()
        self.optimizer.wrapped_optimizer = self.mock_wrapped_optimizer
        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)


class TestDeepSpeedZeroOptimizerStage3Mon(unittest.TestCase):
    def test_get_param_index(self):
        OptimizerMon.wrapped_optimizer = Mock()
        OptimizerMon.wrapped_optimizer.fp16_partitioned_groups = [
            [Mock(flatten=lambda: [1, 2, 3]),
             Mock(flatten=lambda: [4, 5])],
            [Mock(flatten=lambda: [6, 7, 8, 9])]
        ]
        self.params2name = {'param1': 'weight1', 'param2': 'weight2'}
        self.name2index = {'weight1': 0, 'weight2': 2}

        optimizer_stage3_mon = DeepSpeedZeroOptimizerStage3Mon()
        name2indices = optimizer_stage3_mon.get_param_index(self.params2name, self.name2index)

        expected_name2indices = {'weight1': (0, 3, 0, None), 'weight2': (5, 9, 1, None)}
        self.assertDictEqual(dict(name2indices), expected_name2indices)

    def test_fetch_mv(self):
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        OptimizerMon.wrapped_optimizer = Mock()
        OptimizerMon.wrapped_optimizer.fp16_partitioned_groups = MagicMock()

        self.optimizer = DeepSpeedZeroOptimizerStage3Mon()

        # mock _fetch_mv_grad_in_adam
        mv_result = MVGradResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={}, grad={})
        self.mock_fetch_mv_grad_in_adam = MagicMock(return_value=mv_result)
        self.optimizer._fetch_mv_grad_in_adam = self.mock_fetch_mv_grad_in_adam

        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVGradResult)


class TestDeepSpeedZeroOptimizerStage1or2Mon(unittest.TestCase):

    def test_get_group_index(self):
        self.fp32_length = [10, 20, 30, 40]
        self.world_size = 4
        self.indexes = [5, 7, 12, 25, 35, 45]
        self.expected_results = [(40, 0), (40, 0), (12, 1), (24, 2), (34, 2), (40, 0)]

        optimizer = DeepSpeedZeroOptimizerStage1or2Mon()
        results = [optimizer.get_group_index(self.fp32_length, self.world_size, index) for index in self.indexes]
        self.assertEqual(results, self.expected_results)

    @patch('msprobe.pytorch.monitor.optimizer_collect.dist')
    def test_get_param_index(self, mock_dist):
        mock_dist.get_world_size.return_value = 4

        self.params2name = {'param1': 'weight', 'param2': 'bias'}
        self.name2index = {'weight': 0, 'bias': 1}

        self.optimizer_monitor = DeepSpeedZeroOptimizerStage1or2Mon()

        OptimizerMon.wrapped_optimizer = MagicMock()
        OptimizerMon.wrapped_optimizer.groups_padding = [1, 2, 3]
        OptimizerMon.wrapped_optimizer.single_partition_of_fp32_groups = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        OptimizerMon.wrapped_optimizer.bit16_groups = [
            [torch.tensor([6, 7]), torch.tensor([8])],
            [torch.tensor([9, 10, 11])]
        ]

        name2indices = self.optimizer_monitor.get_param_index(self.params2name, self.name2index)
        for name, indices in name2indices.items():
            self.assertIn(name, self.params2name.values())
            self.assertIsInstance(indices, tuple)
            self.assertEqual(len(indices), 4)

    def test_fetch_mv(self):
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        OptimizerMon.wrapped_optimizer = Mock()
        OptimizerMon.wrapped_optimizer.fp16_partitioned_groups = MagicMock()

        self.optimizer = DeepSpeedZeroOptimizerStage1or2Mon()

        # mock _fetch_mv_grad_in_adam
        mv_result = MVGradResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={}, grad={})
        self.mock_fetch_mv_grad_in_adam = MagicMock(return_value=mv_result)
        self.optimizer._fetch_mv_grad_in_adam = self.mock_fetch_mv_grad_in_adam

        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVGradResult)


class TestOptimizerMonFactory(unittest.TestCase):

    def test_create_optimizer_mon(self):
        # 测试已知的优化器类型
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon("Megatron_Float16OptimizerWithFloat16Params"),
                              MixPrecisionOptimizerMon)
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon("Megatron_DistributedOptimizer"),
                              MegatronDistributedOptimizerMon)
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon("Megatron_FP32Optimizer"),
                              MegatronFP32OptimizerMon)
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon("DeepSpeedZeroOptimizer_Stage0"),
                              DeepSpeedZeroOptimizerStage0Mon)
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon("DeepSpeedZeroOptimizer_Stage1_or_2"),
                              DeepSpeedZeroOptimizerStage1or2Mon)
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon("DeepSpeedZeroOptimizer_Stage3"),
                              DeepSpeedZeroOptimizerStage3Mon)

        # 测试未知的优化器类型，应该返回DummyOptimizerMon
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon("unknown"), DummyOptimizerMon)

        # 测试空的优化器类型，应该返回DummyOptimizerMon
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(""), DummyOptimizerMon)

        # 测试异常情况，如果输入的优化器类型不在已知类型列表中，应该抛出异常
        with self.assertRaises(Exception):
            OptimizerMonFactory.create_optimizer_mon("nonexistent")


if __name__ == '__main__':
    unittest.main()
