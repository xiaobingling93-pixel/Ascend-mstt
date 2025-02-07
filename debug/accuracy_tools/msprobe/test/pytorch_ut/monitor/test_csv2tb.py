import os
import shutil
import random
import unittest
import pytest
import torch
import numpy as np
import torch.nn as nn
from msprobe.pytorch.monitor.module_hook import TrainerMon
from msprobe.pytorch import TrainerMon
from msprobe.core.common.const import MonitorConst
from msprobe.pytorch.monitor.csv2tb import parse_step_fn


base_dir = os.path.dirname(os.path.realpath(__file__))
config_json_path = os.path.join(base_dir, "config", "all_config.json")
monitor_output = os.path.join(base_dir, "./monitor_output")
os.environ[MonitorConst.MONITOR_OUTPUT_DIR] = monitor_output


def seed_all(seed=1234, mode=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)

seed_all()


inputs = [torch.rand(10, 10) for _ in range(10)]
labels = [torch.randint(0, 5, (10,)) for _ in range(10)]


class MockModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.relu(x1)
        return x2


def data_collect():
    loss_fun = nn.CrossEntropyLoss()
    test_module = MockModule()
    nn.init.constant_(test_module.linear.weight, 1.0)
    nn.init.constant_(test_module.linear.bias, 1.0)
    optimizer = torch.optim.Adam(test_module.parameters())

    monitor = TrainerMon(config_json_path, params_have_main_grad=False, opt_ty="unknown")
    monitor.monitor_gnorm_with_ad(test_module, grad_acc_steps=1, optimizer=optimizer)

    for input_data, label in zip(inputs, labels):
        output = test_module(input_data)
        loss = loss_fun(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@pytest.fixture(scope="session")
def setup_all():
    data_collect()
    yield
    shutil.rmtree(monitor_output)

@pytest.mark.usefixtures("setup_all")
class TestGradMonitor(unittest.TestCase):
    
    def setUp(self):
        self.maxDiff = None

    def test_actv(self):
        timestamp_dirpath = os.path.join(monitor_output, os.listdir(monitor_output)[0])
        data = parse_step_fn(os.path.join(timestamp_dirpath,"actv_0-2.csv"))
        result = {
            'vp0:.input:micro0': {
                0: {'nans': 0.0,'norm': 5.550016},
                1: {'nans': 0.0,'norm': 5.975112},
                2: {'nans': 0.0,'norm': 5.789881}
                },
            'vp0:.output:micro0': {
                0: {'nans': 0.0,'norm': 41.842655},
                1: {'nans': 0.0,'norm': 44.40981},
                2: {'nans': 0.0,'norm': 43.578354}
                },
            'vp0:linear.input:micro0': {
                0: {'nans': 0.0,'norm': 5.550016},
                1: {'nans': 0.0,'norm': 5.975112},
                2: {'nans': 0.0,'norm': 5.789881}
                },
            'vp0:linear.output:micro0': {
                0: {'nans': 0.0,'norm': 41.842655},
                1: {'nans': 0.0,'norm': 44.40981},
                2: {'nans': 0.0,'norm': 43.578354}
                },
            'vp0:relu.input:micro0': {
                0: {'nans': 0.0,'norm': 41.842655},
                1: {'nans': 0.0,'norm': 44.40981},
                2: {'nans': 0.0,'norm': 43.578354}
                },
            'vp0:relu.output:micro0': {
                0: {'nans': 0.0,'norm': 41.842655},
                1: {'nans': 0.0,'norm': 44.40981},
                2: {'nans': 0.0,'norm': 43.578354}
                }
            }
        self.assertEqual(data, result)
    

    def test_actv_grad(self):
        timestamp_dirpath = os.path.join(monitor_output, os.listdir(monitor_output)[0])
        data = parse_step_fn(os.path.join(timestamp_dirpath,"actv_grad_0-2.csv"))
        nan = np.nan
        result = {
            'vp0:.input:micro0': {
                0: {'norm': nan, 'nans': nan}, 
                1: {'norm': nan, 'nans': nan}, 
                2: {'norm': nan, 'nans': nan}
                }, 
            'vp0:.output:micro0': {
                0: {'norm': 0.282843, 'nans': 0.0}, 
                1: {'norm': 0.282617, 'nans': 0.0}, 
                2: {'norm': 0.282655, 'nans': 0.0}
                }, 
            'vp0:relu.input:micro0': {
                0: {'norm': 0.282843, 'nans': 0.0}, 
                1: {'norm': 0.282617, 'nans': 0.0}, 
                2: {'norm': 0.282655, 'nans': 0.0}
                }, 
            'vp0:relu.output:micro0': {
                0: {'norm': 0.282843, 'nans': 0.0}, 
                1: {'norm': 0.282617, 'nans': 0.0}, 
                2: {'norm': 0.282655, 'nans': 0.0}
                }, 
            'vp0:linear.input:micro0': {
                0: {'norm': nan, 'nans': nan}, 
                1: {'norm': nan, 'nans': nan}, 
                2: {'norm': nan, 'nans': nan}
                },
            'vp0:linear.output:micro0': {
                0: {'norm': 0.282843, 'nans': 0.0}, 
                1: {'norm': 0.282617, 'nans': 0.0}, 
                2: {'norm': 0.282655, 'nans': 0.0}
                }
            }
        
        def dict_equal(a, b):
            if not isinstance(a, dict) or not isinstance(b, dict):
                if np.isnan(a) and np.isnan(b):
                    return True
                return a == b

            if set(a.keys()) != set(b.keys()):
                return False

            for key in a:
                if not dict_equal(a[key], b[key]):
                    return False

            return True
        self.assertEqual(dict_equal(data, result), True)

    
    def test_param(self):
        timestamp_dirpath = os.path.join(monitor_output, os.listdir(monitor_output)[0])
        data = parse_step_fn(os.path.join(timestamp_dirpath,"param_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                0: {'nans': 0.0, 'norm': 2.236068},
                1: {'nans': 0.0, 'norm': 2.236198},
                2: {'nans': 0.0, 'norm': 2.235769}
                },
            'vp0:linear.weight': {
                0: {'nans': 0.0, 'norm': 7.071068},
                1: {'nans': 0.0, 'norm': 7.068808},
                2: {'nans': 0.0, 'norm': 7.06771}
                }
            }
        self.assertEqual(data, result)

    def test_exp_avg(self):
        timestamp_dirpath = os.path.join(monitor_output, os.listdir(monitor_output)[0])
        data = parse_step_fn(os.path.join(timestamp_dirpath,"exp_avg_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                1: {'nans': 0.0, 'norm': 0.024495},
                2: {'nans': 0.0, 'norm': 0.052203}
                },
            'vp0:linear.weight': {
                1: {'nans': 0.0, 'norm': 0.052394},
                2: {'nans': 0.0, 'norm': 0.099221}
                }
            }
        self.assertEqual(data, result)

    def test_exp_avg_sq(self):
        timestamp_dirpath = os.path.join(monitor_output, os.listdir(monitor_output)[0])
        data = parse_step_fn(os.path.join(timestamp_dirpath,"exp_avg_sq_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                1: {'nans': 0.0, 'norm': 4.2e-05},
                2: {'nans': 0.0, 'norm': 9.6e-05}
                },
            'vp0:linear.weight': {
                1: {'nans': 0.0, 'norm': 6.7e-05},
                2: {'nans': 0.0, 'norm': 0.000126}
                }
            }
        self.assertEqual(data, result)
    
    def test_grad_reduced(self):
        timestamp_dirpath = os.path.join(monitor_output, os.listdir(monitor_output)[0])
        data = parse_step_fn(os.path.join(timestamp_dirpath,"grad_reduced_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                0: {'nans': 0.0, 'norm': 0.244949},
                1: {'nans': 0.0, 'norm': 0.314345},
                2: {'nans': 0.0, 'norm': 0.281475}
                },
            'vp0:linear.weight': {
                0: {'nans': 0.0, 'norm': 0.523935},
                1: {'nans': 0.0, 'norm': 0.595672},
                2: {'nans': 0.0, 'norm': 0.497603}
                }
            }
        self.assertEqual(data, result)
        
    def test_grad_unreduced(self):
        timestamp_dirpath = os.path.join(monitor_output, os.listdir(monitor_output)[0])
        data = parse_step_fn(os.path.join(timestamp_dirpath,"grad_unreduced_0-2.csv"))
        result = {
            'vp0:linear.bias': {
                0: {'nans': 0.0, 'norm': 0.244949},
                1: {'nans': 0.0, 'norm': 0.314345},
                2: {'nans': 0.0, 'norm': 0.281475}
                },
            'vp0:linear.weight': {
                0: {'nans': 0.0, 'norm': 0.523935},
                1: {'nans': 0.0, 'norm': 0.595672},
                2: {'nans': 0.0, 'norm': 0.497603}
                }
            }
        self.assertEqual(data, result)

    

