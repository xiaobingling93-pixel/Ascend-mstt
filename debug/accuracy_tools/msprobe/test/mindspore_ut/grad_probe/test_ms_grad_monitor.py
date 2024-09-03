import os
import numpy as np
import shutil
import json
from unittest import TestCase
import hashlib
import mindspore
from mindspore import nn, Tensor
from mindspore.nn import SGD
from msprobe.mindspore import PrecisionDebugger


file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)
config_json_path = os.path.join(directory, "config.json")

def main():
    PrecisionDebugger._instance = None
    debugger = PrecisionDebugger(config_json_path)

    class SimpleNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.my_dense = nn.Dense(16, 5)

        def construct(self, x):
            x = self.flatten(x)
            logits = self.my_dense(x)
            return logits
    model = SimpleNet()
    optimizer = SGD(model.trainable_params(), learning_rate=0.001)

    debugger.monitor(optimizer)

    fix_gradient = tuple([Tensor(np.arange(5*16).reshape((5, 16)), dtype=mindspore.float32),
                        Tensor(np.arange(5).reshape(5), dtype=mindspore.float32)])

    steps = 10

    for _ in range(steps):
        optimizer(fix_gradient)


def save_dict_as_json(data, json_file_path):
    with open(json_file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"字典已保存为json文件: {json_file_path}")


def get_hash(file_path):
    with open(file_path, 'rb') as file:
        hash_object = hashlib.md5()
        for chunk in iter(lambda: file.read(4096), b""):
            hash_object.update(chunk)
    return hash_object.hexdigest()


class TestMsGradientMonitor(TestCase):
    def test_gradient_monitor(self):
        gradient_output_path = os.path.join(directory, "gradient_output")
        config_dict = {
            "task": "grad_probe",
            "dump_path": gradient_output_path,
            "rank": [],
            "step": [1],
            "grad_probe": {
                "grad_level": "L2",
                "param_list": []
            }
        }
        save_dict_as_json(config_dict, config_json_path)

        main()

        my_dense_bias_path = os.path.join(gradient_output_path, "rank0", "step1", "my_dense.bias.npy")
        self.assertTrue(os.path.isfile(my_dense_bias_path), "bias npy file not found")
        my_dense_bias_real = np.load(my_dense_bias_path)
        my_dense_bias_target = np.arange(5).reshape(5) > 0

        self.assertTrue((my_dense_bias_real == my_dense_bias_target).all(), "bias ndarray not same as target")

        my_dense_weight_path = os.path.join(gradient_output_path, "rank0", "step1", "my_dense.weight.npy")
        self.assertTrue(os.path.isfile(my_dense_weight_path), "weight npy file not found")
        my_dense_weight_real = np.load(my_dense_weight_path)
        my_dense_weight_target = np.arange(5*16).reshape((5, 16)) > 0

        self.assertTrue((my_dense_weight_real == my_dense_weight_target).all(), "weight ndarray not same as target")

        real_md5_value = get_hash(os.path.join(gradient_output_path, "rank0", "grad_summary_1.csv"))
        target_md5_value = "874174395c56922f86118050e8c93e74"
        self.assertTrue(real_md5_value, target_md5_value, "hash value of grad_summary_1.csv is not same as target")

        os.remove(config_json_path)
        shutil.rmtree(gradient_output_path)