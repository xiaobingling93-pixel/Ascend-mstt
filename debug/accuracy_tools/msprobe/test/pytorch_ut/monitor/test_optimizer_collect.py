import unittest
from collections import defaultdict
from unittest.mock import Mock, patch, MagicMock

import torch
from msprobe.pytorch.monitor.optimizer_collect import OptimizerMon, \
    OptimizerMonFactory, DummyOptimizerMon, \
    MixPrecisionOptimizerMon, MegatronDistributedOptimizerMon, MegatronFP32OptimizerMon, \
    MegatronChainedDistributedOptimizerMon, MegatronChainedMixPrecisionOptimizerMon, \
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
        self.torch_opt.state = defaultdict(dict)
        self.torch_opt.averaged_gradients = defaultdict(torch.Tensor)
        self.torch_opt.partition_size = defaultdict(int)
        self.torch_opt.flatten_dense_tensors_aligned = MagicMock()
        self.torch_opt.flatten = MagicMock()

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
        self.optimizer.fp16_to_fp32_param = {}

        # Mock _fetch_mv_in_adam method and set a fixed return value
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv_in_adam = MagicMock(return_value=mv_result)
        self.optimizer._fetch_mv_in_adam = self.mock_fetch_mv_in_adam

        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.mock_fetch_mv_in_adam.assert_called_once_with(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)


class TestChainedMixPrecisionOptimizerMon(unittest.TestCase):
    def test_fetch_mv_with_fp16_to_fp32_param_and_mix_prec_opt(self):
        # init monitor, torch_opt ...
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        self.torch_opt.float16_groups = [MagicMock()]
        self.torch_opt.fp32_from_float16_groups = [MagicMock()]
        self.optimizer = MegatronChainedMixPrecisionOptimizerMon()
        self.optimizer.optimizer = [MagicMock(), MagicMock()]
        self.optimizer.fp16_to_fp32_param = {}

        # Mock _fetch_mv_in_adam method and set a fixed return value
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv_in_adam = MagicMock(return_value=mv_result)
        self.optimizer._fetch_mv_in_adam = self.mock_fetch_mv_in_adam

        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.mock_fetch_mv_in_adam.assert_called_once_with(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)


class TestMegatronChainedDistributedOptimizerMon(unittest.TestCase):
    def setUp(self):
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv_in_adam = MagicMock(return_value=mv_result)
        self.optimizer = MegatronChainedDistributedOptimizerMon()

    def test_fetch_mv_with_valid_optimizer(self):
        self.torch_opt.model_float16_groups = [MagicMock()]
        self.torch_opt.shard_fp32_from_float16_groups = [MagicMock()]
        self.optimizer._fetch_mv_in_adam = self.mock_fetch_mv_in_adam

        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)

    def test_fetch_mv_with_invalid_optimizer(self):
        self.torch_opt = Mock()
        self.torch_opt.model_float16_groups = None
        self.torch_opt.shard_fp32_from_float16_groups = None
        self.optimizer._fetch_mv_in_adam = self.mock_fetch_mv_in_adam

        with self.assertRaises(Exception):
            self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)


class TestMegatronDistributedOptimizerMon(unittest.TestCase):
    def setUp(self):
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        mv_result = MVResult(exp_avg={}, exp_avg_sq={}, update={}, ratio={})
        self.mock_fetch_mv_in_adam = MagicMock(return_value=mv_result)
        self.optimizer = MegatronDistributedOptimizerMon()

    def test_fetch_mv_with_valid_optimizer(self):
        self.torch_opt.model_float16_groups = [MagicMock()]
        self.torch_opt.shard_fp32_from_float16_groups = [MagicMock()]
        self.optimizer._fetch_mv_in_adam = self.mock_fetch_mv_in_adam

        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)

    def test_fetch_mv_with_invalid_optimizer(self):
        self.torch_opt = Mock()
        self.torch_opt.model_float16_groups = None
        self.torch_opt.shard_fp32_from_float16_groups = None
        self.optimizer._fetch_mv_in_adam = self.mock_fetch_mv_in_adam

        with self.assertRaises(Exception):
            self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)


class TestCommonFetchMv(unittest.TestCase):
    def setUp(self) -> None:
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()

    def test_megatron_fp32_optimizer_mon(self):
        self.optimizer = MegatronFP32OptimizerMon()
        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)

    def test_deepspeed_zero_optimizer_stage0_mon(self):
        self.optimizer = DeepSpeedZeroOptimizerStage0Mon()
        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)

    def test_dummy_optimizer_mon(self):
        self.optimizer = DummyOptimizerMon()
        res = self.optimizer.fetch_mv(self.monitor, self.torch_opt, self.params2name)
        self.assertIsInstance(res, MVResult)


class TestDeepSpeedZeroOptimizerStage3Mon(unittest.TestCase):
    def test_get_param_index(self):
        self.torch_opt = Mock()
        self.torch_opt.fp16_partitioned_groups = [
            [Mock(flatten=lambda: [1, 2, 3]),
             Mock(flatten=lambda: [4, 5])],
            [Mock(flatten=lambda: [6, 7, 8, 9])]
        ]
        self.params2name = {'param1': 'weight1', 'param2': 'weight2'}
        self.name2index = {'weight1': 0, 'weight2': 2}

        optimizer_stage3_mon = DeepSpeedZeroOptimizerStage3Mon()
        name2indices = optimizer_stage3_mon.get_param_index(self.params2name, self.name2index, self.torch_opt)

        expected_name2indices = {'weight1': (0, 3, 0, None), 'weight2': (5, 9, 1, None)}
        self.assertDictEqual(dict(name2indices), expected_name2indices)

    def test_fetch_mv(self):
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        self.torch_opt.fp16_partitioned_groups = MagicMock()
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

        self.torch_opt = MagicMock()
        self.torch_opt.groups_padding = [1, 2, 3]
        self.torch_opt.single_partition_of_fp32_groups = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        self.torch_opt.bit16_groups = [
            [torch.tensor([6, 7]), torch.tensor([8])],
            [torch.tensor([9, 10, 11])]
        ]

        name2indices = self.optimizer_monitor.get_param_index(self.params2name, self.name2index, self.torch_opt)
        for name, indices in name2indices.items():
            self.assertIn(name, self.params2name.values())
            self.assertIsInstance(indices, tuple)
            self.assertEqual(len(indices), 4)

    def test_fetch_mv(self):
        self.monitor = MagicMock()
        self.torch_opt = MagicMock()
        self.params2name = MagicMock()
        self.torch_opt.fp16_partitioned_groups = MagicMock()
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
        mix_optimizer = MagicMock()
        mix_optimizer_class = MagicMock()
        mix_optimizer_class.__name__ = "Float16OptimizerWithFloat16Params"
        mix_optimizer.__class__ = mix_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(mix_optimizer)[0],
                              MixPrecisionOptimizerMon)
        dis_optimizer = MagicMock()
        dis_optimizer_class = MagicMock()
        dis_optimizer_class.__name__ = "DistributedOptimizer"
        dis_optimizer.__class__ = dis_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(dis_optimizer)[0],
                              MegatronDistributedOptimizerMon)
        fp32_optimizer = MagicMock()
        fp32_optimizer_class = MagicMock()
        fp32_optimizer_class.__name__ = "FP32Optimizer"
        fp32_optimizer.__class__ = fp32_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(fp32_optimizer)[0],
                              MegatronFP32OptimizerMon)
        chained_optimizer = MagicMock()
        chained_optimizer_class = MagicMock()
        chained_optimizer_class.__name__ = "ChainedOptimizer"
        chained_optimizer.__class__ = fp32_optimizer_class
        chained_optimizer.chained_optimizers = [mix_optimizer, mix_optimizer]
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(chained_optimizer)[0],
                              MegatronChainedDistributedOptimizerMon)
        chained_optimizer.chained_optimizers = [dis_optimizer, dis_optimizer]
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(chained_optimizer)[0],
                              MegatronChainedMixPrecisionOptimizerMon)
        deepspeed_optimizer = MagicMock()
        deepspeed_optimizer_class = MagicMock()
        deepspeed_optimizer_class.__name__ = "BF16_Optimizer"
        deepspeed_optimizer.__class__ = deepspeed_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(deepspeed_optimizer)[0],
                              DeepSpeedZeroOptimizerStage0Mon)
        deepspeed_optimizer_class.__name__ = "DeepSpeedZeroOptimizer"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(deepspeed_optimizer)[0],
                              DeepSpeedZeroOptimizerStage1or2Mon)
        deepspeed_optimizer_class.__name__ = "DeepSpeedZeroOptimizer_Stage3"
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(deepspeed_optimizer)[0],
                              DeepSpeedZeroOptimizerStage3Mon)
        # 测试未知的优化器类型，应该返回DummyOptimizerMon
        unknow_optimizer = MagicMock()
        unknow_optimizer_class = MagicMock()
        unknow_optimizer_class.__name__ = "unknown"
        unknow_optimizer.__class__ = unknow_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(unknow_optimizer)[0], DummyOptimizerMon)


if __name__ == '__main__':
    unittest.main()
