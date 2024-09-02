import os
import numpy as np
import shutil
import yaml
from unittest import TestCase
import hashlib
from grad_tool.test.ut_ms.only_optimizer import main


file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)


def save_dict_as_yaml(data, yaml_file_path):
    # 将字典保存为YAML文件
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    print(f"字典已保存为YAML文件: {yaml_file_path}")


def get_hash(file_path):
    with open(file_path, 'rb') as file:
        hash_object = hashlib.md5()
        for chunk in iter(lambda: file.read(4096), b""):
            hash_object.update(chunk)
    return hash_object.hexdigest()


def check_npy(gradient_output_path):
    my_dense_bias_path = os.path.join(gradient_output_path, "rank_0", "step_1", "my_dense.bias.npy")
    assert  os.path.isfile(my_dense_bias_path)
    my_dense_bias_real = np.load(my_dense_bias_path)
    my_dense_bias_target = np.arange(5).reshape(5) > 0

    assert (my_dense_bias_real == my_dense_bias_target).all()

    my_dense_weight_path = os.path.join(gradient_output_path, "rank_0", "step_1", "my_dense.weight.npy")
    assert  os.path.isfile(my_dense_weight_path)
    my_dense_weight_real = np.load(my_dense_weight_path)
    my_dense_weight_target = np.arange(5*16).reshape((5, 16)) > 0

    assert (my_dense_weight_real == my_dense_weight_target).all()

def check_stat_csv(csv_path, md5_value):
    real_md5_value = get_hash(csv_path)
    assert real_md5_value == md5_value


class TestMsGradientMonitor(TestCase):
    def test_gradient_monitor(self):
        config_path = os.path.join(directory, "config.yaml")
        gradient_output_path = os.path.join(directory, "gradient_output")
        config_dict = {
            "level": "L2",
            "param_list": None,
            "rank": None,
            "step": [1],
            "bounds": None,
            "output_path": gradient_output_path,
        }
        save_dict_as_yaml(config_dict, config_path)

        main()
        check_npy(gradient_output_path)
        check_stat_csv(os.path.join(gradient_output_path, "rank_0", "grad_summary_1.csv"), "b00499bee5c3e3ca96aaee773a092dd7")
        os.remove(config_path)
        shutil.rmtree(gradient_output_path)