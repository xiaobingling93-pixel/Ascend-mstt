import os
import glob
import re
import time
import shutil
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, grad
from msprobe.mindspore import save, save_grad


@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={'jit_level': "O2"}, device_target="Ascend")
    dump_dir = "./test_dump"
    os.makedirs(dump_dir, exist_ok=True)
    
    yield dump_dir  # This is where the test runs
    
    # Teardown
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)


def test_save_when_net_then_dump_data_in_dir(setup_teardown):
    """Test forward pass saving"""
    class Net(nn.Cell):
        def construct(self, x, y, z):
            save("./test_dump", 'x', x)
            return x * y * z

    x = Tensor([1, 2], ms.float32)
    y = Tensor([-2, 3], ms.float32)
    z = Tensor([0, 3], ms.float32)
    net = Net()
    output = grad(net, grad_position=(1, 2))(x, y, z)
    time.sleep(1)

    # Verify file generation
    step_dir = os.path.join("./test_dump", "step0", "rank0")
    file_pattern = re.compile(r'x_float32_\d+\.npy')
    files = os.listdir(step_dir)
    matched_files = [f for f in files if file_pattern.match(f)]
    assert len(matched_files) > 0, "Expected at least 1 files matching the pattern"

def test_save_grad_when_net_then_dump_in_dir(setup_teardown):
    """Test gradient saving"""
    class Net(nn.Cell):
        def construct(self, x, y, z):
            z = save_grad("./test_dump", 'z', z)
            return x * y * z

    x = Tensor([1, 2], ms.float32)
    y = Tensor([-2, 3], ms.float32)
    z = Tensor([0, 3], ms.float32)
    net = Net()
    output = grad(net, grad_position=(1, 2))(x, y, z)
    time.sleep(1)

    # Verify file generation
    step_dir = os.path.join("./test_dump", "step0", "rank0")
    file_pattern = re.compile(r'z_grad_float32_\d+\.npy')
    files = os.listdir(step_dir)
    matched_files = [f for f in files if file_pattern.match(f)]
    assert len(matched_files) > 0, "Expected at least 1 files matching the pattern"