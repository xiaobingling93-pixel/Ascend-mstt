import unittest
from collections import defaultdict
from unittest.mock import Mock, patch, MagicMock

import torch
from torch._utils import _flatten_dense_tensors
from msprobe.pytorch.monitor.optimizer_collect import OptimizerMon, \
    OptimizerMonFactory, DummyOptimizerMon, \
    MixPrecisionOptimizerMon, MegatronDistributedOptimizerMon, MegatronFP32OptimizerMon, \
    MegatronChainedDistributedOptimizerMon, MegatronChainedMixPrecisionOptimizerMon, \
    DeepSpeedZeroOptimizerStage0Mon, DeepSpeedZeroOptimizerStage1or2Mon, DeepSpeedZeroOptimizerStage3Mon

from msprobe.pytorch.monitor.utils import MVResult, MVGradResult


def setup_param_groups(num_groups=2, params_per_group=5):
    bit16_groups = []
    param_names = {}
    name2index = {}
    param_slice_mappings = []
    count = 0
    for group_idx in range(num_groups):
        group = []
        param_slice_mapping = {}
        offset = 0
        for i in range(params_per_group):
            name = f'param{group_idx}_{i}'
            p = torch.nn.Parameter(torch.randn(2,3, dtype=torch.bfloat16))
            p.ds_tensor = torch.nn.Parameter(torch.randn(1,3, dtype=torch.bfloat16))
            param_slice_mapping[name] = MagicMock(start=offset, numel=p.numel())
            name2index[name] = count
            group.append(p)
            param_names[p] = name
            offset += p.numel()
            count += 1
        bit16_groups.append(group)
        param_slice_mappings.append(param_slice_mapping)
    
    return  bit16_groups, param_names, name2index, param_slice_mappings

def setup_mock_monitor():
    mock_monitor = MagicMock()
    mock_monitor.mv_distribution = True
    mock_monitor.mg_direction = False
    mock_monitor.ur_distribution = False

    return mock_monitor

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
    def setUp(self):
        bit16_groups, param_names, name2index, _ = setup_param_groups()

        mock_opt = MagicMock()
        mock_opt.param_names = param_names
        mock_opt.fp16_groups = bit16_groups
        mock_opt.fp32_partitioned_groups_flat = [torch.stack(group,dim=0).flatten().float() \
                                                 for group in bit16_groups]
        mock_opt.fp16_partitioned_groups = [[p.ds_tensor for p in group] for  group in bit16_groups]        
        mock_opt.flatten = _flatten_dense_tensors
        mock_opt.averaged_gradients = {group_idx: [torch.randn_like(param.ds_tensor) for param in group] for group_idx, group in enumerate(bit16_groups)}
        mock_opt.state = {
            flat_group: {
                'exp_avg': torch.ones_like(flat_group),
                'exp_avg_sq': torch.ones_like(flat_group)
            } for flat_group in mock_opt.fp32_partitioned_groups_flat
        } 

        self.torch_opt = mock_opt
        self.optimizer_mon = DeepSpeedZeroOptimizerStage3Mon()
        self.mock_monitor = setup_mock_monitor()
        self.name2index = name2index
        self.params2name = param_names
        
    def test_get_param_index(self):
        name2indices = self.optimizer_mon.get_param_index(self.params2name, self.name2index, self.torch_opt)
        expected_name2indices = {
            'param0_0': (0, 3, 0, None),
            'param0_1': (3, 6, 0, None),
            'param0_2': (6, 9, 0, None),
            'param0_3': (9, 12, 0, None),
            'param0_4': (12, 15, 0, None),
            'param1_0': (0, 3, 1, None),
            'param1_1': (3, 6, 1, None),
            'param1_2': (6, 9, 1, None),
            'param1_3': (9, 12, 1, None),
            'param1_4': (12, 15, 1, None)
        }
        self.assertDictEqual(name2indices, expected_name2indices)

    def test_fetch_mv(self):
        name2indices = self.optimizer_mon.get_param_index(self.params2name, self.name2index, self.torch_opt)
        result = self.optimizer_mon.fetch_mv(self.mock_monitor, self.torch_opt, self.params2name, name2indices)

        for param, name in self.torch_opt.param_names.items():  
            self.assertTrue(torch.equal(result.exp_avg[name], torch.ones_like(param.ds_tensor).flatten()))
            self.assertTrue(torch.equal(result.exp_avg_sq[name], torch.ones_like(param.ds_tensor).flatten()))


class TestDeepSpeedZeroOptimizerStage1or2Mon(unittest.TestCase):
    def setUp(self):
        """Mock zero1/2 partitions
        """
        bit16_groups, param_names, name2index, _ = setup_param_groups()
        mock_opt = MagicMock()
        mock_opt.groups_padding = [0, 0]
        mock_opt.single_partition_of_fp32_groups = [torch.stack(group,dim=0).flatten() for group in bit16_groups]
        mock_opt.partition_size = [p.numel() for p in mock_opt.single_partition_of_fp32_groups]
        mock_opt.param_names = param_names
        mock_opt.bit16_groups = bit16_groups
        mock_opt.averaged_gradients = {group_idx: [torch.randn_like(param) for param in group] \
                                       for group_idx, group in enumerate(mock_opt.single_partition_of_fp32_groups)}

        mock_opt.flatten = _flatten_dense_tensors
        def flatten_dense_tensors_aligned(tensor_list, alignment):
            return _flatten_dense_tensors(tensor_list)
        mock_opt.flatten_dense_tensors_aligned = flatten_dense_tensors_aligned

        mock_opt.state = {
            flat_group: {
                'exp_avg': torch.ones_like(flat_group),
                'exp_avg_sq': torch.ones_like(flat_group)
            } for flat_group in mock_opt.single_partition_of_fp32_groups
        } 

        self.torch_opt = mock_opt
        self.optimizer_mon = DeepSpeedZeroOptimizerStage1or2Mon()
        self.mock_monitor = setup_mock_monitor()
        self.name2index = name2index
        self.params2name = param_names

    def test_get_group_index(self):
        self.fp32_length = [10, 20, 30, 40]
        self.world_size = 4
        self.indexes = [5, 7, 12, 25, 35, 45]
        self.expected_results = [(40, 0), (40, 0), (12, 1), (24, 2), (34, 2), (40, 0)]

        results = [self.optimizer_mon.get_group_index(self.fp32_length, self.world_size, index) for index in self.indexes]
        self.assertEqual(results, self.expected_results)

    def test_get_param_index(self):
        name2indices = self.optimizer_mon.get_param_index(self.params2name, self.name2index, self.torch_opt)
        for name, indices in name2indices.items():
            self.assertIn(name, self.params2name.values())
            self.assertIsInstance(indices, tuple)
            self.assertEqual(len(indices), 4)

    def test_fetch_mv(self):
        # mock _fetch_mv_grad_in_adam
        name2indices = self.optimizer_mon.get_param_index(self.params2name, self.name2index, self.torch_opt)
        result = self.optimizer_mon.fetch_mv(self.mock_monitor, self.torch_opt, self.params2name, name2indices)

        for param, name in self.torch_opt.param_names.items():  
            self.assertTrue(torch.equal(result.exp_avg[name], torch.ones_like(param).flatten()))
            self.assertTrue(torch.equal(result.exp_avg_sq[name], torch.ones_like(param).flatten()))


class TestDeepSpeedZeroOptimizerStage0Mon(unittest.TestCase):
    def setUp(self):
        bit16_groups, param_names, name2index, param_slice_mapping = setup_param_groups()
        mock_opt = MagicMock()

        mock_opt.bf16_groups = bit16_groups
        mock_opt.fp32_groups_flat_partition = [torch.stack(group,dim=0).flatten() for group in bit16_groups]
        mock_opt.optimizer.state = {
            flat_group: {
                'exp_avg': torch.ones_like(flat_group),
                'exp_avg_sq': torch.ones_like(flat_group)
            } for flat_group in mock_opt.fp32_groups_flat_partition
        } 

        mock_opt.state_dict.return_value = {'param_slice_mappings':param_slice_mapping}
        mock_opt.param_names = param_names

        self.torch_opt = mock_opt
        self.optimizer_mon = DeepSpeedZeroOptimizerStage0Mon()
        self.mock_monitor = setup_mock_monitor()
        self.name2index = name2index
        self.params2name = param_names

    def test_fetch_mv(self):
        result = self.optimizer_mon.fetch_mv(self.mock_monitor, self.torch_opt, self.params2name)

        for param, name in self.torch_opt.param_names.items():  
            self.assertTrue(torch.equal(result.exp_avg[name], torch.ones_like(param).flatten()))
            self.assertTrue(torch.equal(result.exp_avg_sq[name], torch.ones_like(param).flatten()))


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
        chained_optimizer.__class__ = chained_optimizer_class
        chained_optimizer.chained_optimizers = [mix_optimizer, mix_optimizer]
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(chained_optimizer)[0],
                              MegatronChainedMixPrecisionOptimizerMon)
        chained_optimizer.chained_optimizers = [dis_optimizer, dis_optimizer]
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(chained_optimizer)[0],
                              MegatronChainedDistributedOptimizerMon)
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
        unknown_optimizer = MagicMock()
        unknown_optimizer_class = MagicMock()
        unknown_optimizer_class.__name__ = "unknown"
        unknown_optimizer.__class__ = unknown_optimizer_class
        self.assertIsInstance(OptimizerMonFactory.create_optimizer_mon(unknown_optimizer)[0], DummyOptimizerMon)


if __name__ == '__main__':
    unittest.main()
