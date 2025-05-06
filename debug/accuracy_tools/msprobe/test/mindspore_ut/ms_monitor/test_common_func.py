import pytest
from mindspore import nn, context
from mindspore.common.initializer import Normal
import mindspore as ms
import numpy as np

from msprobe.mindspore.common.utils import is_mindtorch, mindtorch_check_result
from msprobe.mindspore.monitor.common_func import (
    is_valid_instance,
    get_submodules,
    get_parameters,
    get_rank,
    comm_is_initialized,
    optimizer_pre_hook,
    optimizer_post_hook
)

mindtorch_check_result = None
TORCH_AVAILABLE = False
if is_mindtorch():
    try:
        import torch
        import torch.nn as torch_nn
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False


class TestModelUtils:
    @classmethod
    def setup_class(cls):
        """Setup once for all tests in this class"""
        cls.ms_model = MSModel()
        if TORCH_AVAILABLE:
            cls.torch_model = TorchModel()

    @classmethod
    def teardown_class(cls):
        """Cleanup after all tests in this class"""
        pass

    @pytest.fixture(params=['torch'] if TORCH_AVAILABLE else ['mindspore'])
    def model(self, request):
        """Fixture providing both MindSpore and PyTorch models"""
        if request.param == 'mindspore':
            return self.ms_model
        elif request.param == 'torch' and TORCH_AVAILABLE:
            return self.torch_model

    def test_is_valid_instance_if_model_is_cell_or_module_then_return_true(self, model):
        assert is_valid_instance(model)

    def test_is_valid_instance_if_input_is_string_then_return_false(self):
        assert not is_valid_instance("not a model")

    def test_is_valid_instance_if_input_is_number_then_return_false(self):
        assert not is_valid_instance(123)

    def test_get_submodules_if_model_is_valid_then_return_non_empty_dict(self, model):
        submodules = dict(get_submodules(model))
        assert len(submodules) > 0
        if isinstance(model, nn.Cell):
            assert any(name.endswith('conv1') for name in submodules)
        elif TORCH_AVAILABLE and isinstance(model, torch_nn.Module):
            assert any(name == 'conv1' for name in submodules)

    def test_get_submodules_if_model_is_invalid_then_return_empty_dict(self):
        assert get_submodules("invalid") == {}

    def test_get_parameters_if_model_is_valid_then_return_non_empty_dict(self, model):
        params = dict(get_parameters(model))
        assert len(params) > 0
        if isinstance(model, nn.Cell):
            assert any('conv1.weight' in name for name in params)
        elif TORCH_AVAILABLE and isinstance(model, torch_nn.Module):
            assert any(name == 'conv1.weight' for name in params)

    def test_get_parameters_if_model_is_invalid_then_return_empty_dict(self):
        assert get_parameters(123) == {}

    def test_get_rank_if_comm_initialized_then_return_integer(self):
        rank = get_rank()
        assert isinstance(rank, int)
        assert rank >= 0

    def test_comm_is_initialized_when_called_then_return_boolean(self):
        assert isinstance(comm_is_initialized(), bool)


# Test models
class MSModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, has_bias=True, weight_init=Normal(0.02))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

if TORCH_AVAILABLE:
    class TorchModel(torch_nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch_nn.Conv2d(3, 64, 3)
            self.bn1 = torch_nn.BatchNorm2d(64)
            self.relu = torch_nn.ReLU()
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return x