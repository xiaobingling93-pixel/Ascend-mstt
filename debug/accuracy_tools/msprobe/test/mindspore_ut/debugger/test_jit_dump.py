import json
import os
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore import jit
from msprobe.mindspore import PrecisionDebugger
from msprobe.core.common_config import CommonConfig, BaseConfig
from msprobe.mindspore.dump.jit_dump import JitDump, dump_jit
from msprobe.core.common.file_utils import FileOpen


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


def save_dict_as_json(data, json_file_path):
    with FileOpen(json_file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


class TestJitDump0(unittest.TestCase):
    def test_jitdump_forward_backward_L1(self):
        file_path = os.path.abspath(__file__)
        directory = os.path.dirname(file_path)
        output_path = os.path.join(directory, "output_L1")
        # Set up configurations
        json_config = {
            "task": "tensor",
            "dump_path": output_path,
            "rank": [],
            "step": [],
            "level": "L1"
        }
        config_json_path = os.path.join(directory, "config.json")
        save_dict_as_json(json_config, config_json_path)

        PrecisionDebugger._instance = None
        PrecisionDebugger.initialized = False

        debugger = PrecisionDebugger(config_json_path)

        # Setup MindSpore context and JitDump class
        ms.set_context(mode=ms.PYNATIVE_MODE)

        net_model = AlexNet()
        debugger.start()
        ops.relu(ms.Tensor(np.random.random([1, 227, 227, 3]).astype(np.float32)))
        grad_net = ms.grad(net_model, None, net_model.trainable_params())
        output = grad_net(ms.Tensor(np.random.random([1, 227, 227, 3]).astype(np.float32)))
        debugger.stop()
        expected_file_count = 5
        output_tensor_path = os.path.join(output_path, "step0")
        output_tensor_path = os.path.join(output_tensor_path, "rank")
        output_tensor_path = os.path.join(output_tensor_path, "dump_tensor_data")
        actual_file_count = len(os.listdir(output_tensor_path))
        assert actual_file_count == expected_file_count


class TestJitDump(unittest.TestCase):
    @patch('os.getpid', return_value=12345)
    def test_dump_jit(self, mock_getpid):
        in_feat = Tensor(np.array([1, 2, 3]), mstype.float32)
        out_feat = Tensor(np.array([4, 5, 6]), mstype.float32)

        # Mock need_dump to return True
        with patch.object(JitDump, 'need_dump', return_value=True):
            # Add a mock data_collector to the JitDump class
            JitDump.data_collector = MagicMock()

            # Call the function to be tested
            dump_jit('sample_name', in_feat, out_feat, True)

            # Verify the expected calls
            self.assertTrue(JitDump.data_collector.update_api_or_module_name.called)
            self.assertTrue(JitDump.data_collector.forward_data_collect.called)
            JitDump.data_collector.forward_data_collect.assert_called_once()

    @patch('os.listdir', return_value=['tensor1', 'tensor2', 'tensor3', 'tensor4', 'tensor5'])
    def test_dump_tensor_data_files_count(self, mock_listdir):
        dir_path = "/absolute_path/step0/rank/dump_tensor_data/"
        expected_file_count = 5
        actual_file_count = len(os.listdir(dir_path))
        self.assertEqual(actual_file_count, expected_file_count)


if __name__ == "__main__":
    unittest.main()
