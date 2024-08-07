import numpy as np
import os
from unittest.mock import patch, MagicMock
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore import jit
from msprobe.mindspore import PrecisionDebugger
from msprobe.core.common_config import CommonConfig, BaseConfig

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
has_bias=has_bias, pad_mode=pad_mode)

def fc_with_initialize(input_channels, out_channels, has_bias=True):
    return nn.Dense(input_channels, out_channels, has_bias=has_bias)

class DataNormTranspose(nn.Cell):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respectively.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respectively.
    """

    def __init__(self, dataset_name='imagenet'):
        super(DataNormTranspose, self).__init__()
        # Computed from random subset of ImageNet training images
        if dataset_name == 'imagenet':
            self.mean = Tensor(np.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).reshape((1, 1, 1, 3)), mstype.float32)
            self.std = Tensor(np.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).reshape((1, 1, 1, 3)), mstype.float32)
        else:
            self.mean = Tensor(np.array([0.4914, 0.4822, 0.4465]).reshape((1, 1, 1, 3)), mstype.float32)
            self.std = Tensor(np.array([0.2023, 0.1994, 0.2010]).reshape((1, 1, 1, 3)), mstype.float32)

    def construct(self, x):
        x = (x - self.mean) / self.std
        x = ops.transpose(x, (0, 3, 1, 2))
        return x

class AlexNet(nn.Cell):
    """
    Alexnet
    """

    def __init__(self, num_classes=10, channel=3, phase='train', include_top=True, dataset_name='imagenet'):
        super(AlexNet, self).__init__()
        self.data_trans = DataNormTranspose(dataset_name=dataset_name)
        self.conv1 = conv(channel, 64, 11, stride=4, pad_mode="same", has_bias=True)
        self.conv2 = conv(64, 128, 5, pad_mode="same", has_bias=True)
        self.conv3 = conv(128, 192, 3, pad_mode="same", has_bias=True)
        self.conv4 = conv(192, 256, 3, pad_mode="same", has_bias=True)
        self.conv5 = conv(256, 256, 3, pad_mode="same", has_bias=True)
        self.relu = nn.ReLU()
        nn.BatchNorm2d
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0
            self.flatten = nn.Flatten()
            self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
            self.fc2 = fc_with_initialize(4096, 4096)
            self.fc3 = fc_with_initialize(4096, num_classes)
            self.dropout = nn.Dropout(p=1 - dropout_ratio)

    @jit
    def construct(self, x):
        """define network"""
        x = self.data_trans(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = ops.celu(x, 2.0)
        return x

if __name__ == "__main__":
    json_config = {
        "task": "statistics",
        "dump_path": "/absolute_path",
        "rank": [],
        "step": [],
        "level": "L1"
    }

    common_config = CommonConfig(json_config)
    task_config = BaseConfig(json_config)
    mock_parse_json_config = MagicMock()
    mock_parse_json_config.return_value = [common_config, task_config]
    debugger = PrecisionDebugger()
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = AlexNet()
    debugger.start()
    ops.relu(ms.Tensor(np.random.random([1, 227, 227, 3]).astype(np.float32)))
    grad_net = ms.grad(net, None, net.trainable_params())
    output = grad_net(ms.Tensor(np.random.random([1, 227, 227, 3]).astype(np.float32)))
    debugger.stop()
    expected_file_count = 5
    dir_path = "/absolute_path/step0/rank/dump_tensor_data/"
    actual_file_count = len(os.listdir(dir_path))
    assert actual_file_count == expected_file_count
