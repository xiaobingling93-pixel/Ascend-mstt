import unittest
import os
import json
import torch
import numpy as np
import shutil

from msprobe.pytorch import PrecisionDebugger

current_file = __file__
parent_dir = os.path.abspath(os.path.dirname(current_file))
test_dir = os.path.join(parent_dir, "test_dir")

def deep_compare(obj1, obj2, float_tolerance=1e-5):
    """
    Recursively compare two objects to check if they are the same.
    Supports nested dictionaries and lists.
    """
    if type(obj1) != type(obj2):
        return False
    if isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(deep_compare(obj1[key], obj2[key]) for key in obj1)
    if isinstance(obj1, (tuple, list)):
        if len(obj1) != len(obj2):
            return False
        return all(deep_compare(item1, item2) for item1, item2 in zip(obj1, obj2))
    if isinstance(obj1, (int, float)):
        return abs(obj1 - obj2) < float_tolerance
    return obj1 == obj2

class TestDebuggerSave(unittest.TestCase):
    @staticmethod
    def write_config_json(step, async_dump, mode, dump_path, config_file_path):
        task = "tensor" if mode == "tensor" else "statistics"
        statistics_summary_mode = "statistics" if mode == "statistics" else "md5"
        config = {
            "task": task,
            "dump_path": dump_path,
            "rank": [],
            "step": step,
            "level": "debug",
            "enable_dataloader": False,
            "async_dump": async_dump,
            "statistics": {
                "summary_mode": statistics_summary_mode,
            }
        }
        with open(config_file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    @staticmethod
    def read_debug_json_into_dict(debug_json_path):
        with open(debug_json_path, "r", encoding="utf-8") as f:
            debug_json = json.load(f)
        return debug_json


    @staticmethod
    def check_real_pt(pt_path, target_pt_tensor, check_values=True, rtol=1e-5, atol=1e-8):
        """
        Enhanced version with optional value comparison.

        Args:
            pt_path (str): Path to the .pt file
            target_pt_tensor: Target torch tensor to compare
            check_values (bool): If True, also compare array values
            rtol, atol: Relative and absolute tolerances for value comparison

        Returns:
            bool: True if all checks pass
        """
        # Load the pt file
        try:
            pt_data = torch.load(pt_path)
        except FileNotFoundError:
            print(f"Error: The file {pt_path} does not exist.")
            return False
        except Exception as e:
            print(f"Error loading pt file: {e}")
            return False
        # Check shapes
        if pt_data.shape != target_pt_tensor.shape:
            print(f"Shape mismatch: pt data shape is {pt_data.shape}, target tensor shape is {target_pt_tensor.shape}")
            return False
        # Check dtypes
        if pt_data.dtype != target_pt_tensor.dtype:
            print(f"Shape mismatch: pt data dtype is {pt_data.dtype}, target tensor dtype is {target_pt_tensor.dtype}")
            return False
        # Optionally check values
        if check_values:
            if not torch.allclose(pt_data, target_pt_tensor, rtol=rtol, atol=atol):
                print("Value mismatch: pt data and target tensor values do not match within the specified tolerances.")
                return False
        return True

    def setUp(self):
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        PrecisionDebugger._instance = None

    def tearDown(self):
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        PrecisionDebugger._instance = None

    def test_save_real_tensor(self):
        data = {"a": torch.Tensor([1., 2.])}
        step = []
        async_dump = False
        mode = "tensor"
        dump_path = os.path.join(test_dir, "debug_save")
        config_file_path = os.path.join(test_dir, "config.json")
        self.write_config_json(step, async_dump, mode, dump_path, config_file_path)
        debugger =  PrecisionDebugger(config_file_path)
        PrecisionDebugger.save(data, "data_dict", save_backward=False)
        PrecisionDebugger.step()
        # check pt file
        pt_path = os.path.join(dump_path, "step0", "rank", "dump_tensor_data", "data_dict.0.debug.a.pt")
        assert self.check_real_pt(pt_path, data["a"])
        # check debug json
        target_debug_info = {
            "a": {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [
                2
                ],
                "Max": 2.0,
                "Min": 1.0,
                "Mean": 1.5,
                "Norm": 2.2360680103302,
                "requires_grad": False,
                "data_name": "data_dict.0.debug.a.pt"
            }
        }
        debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
        debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
        assert deep_compare(debug_json_dict["data"]["data_dict.0.debug"], target_debug_info)

    def test_save_md5(self):
        data = {"a": torch.Tensor([1., 2.])}
        step = []
        async_dump = False
        mode = "md5"
        dump_path = os.path.join(test_dir, "debug_save")
        config_file_path = os.path.join(test_dir, "config.json")
        self.write_config_json(step, async_dump, mode, dump_path, config_file_path)
        debugger =  PrecisionDebugger(config_file_path)
        PrecisionDebugger.save(data, "data_dict", save_backward=False)
        PrecisionDebugger.step()
        # check debug json
        target_debug_info = {
            "a": {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [
                2
                ],
                "Max": 2.0,
                "Min": 1.0,
                "Mean": 1.5,
                "Norm": 2.2360680103302,
                "requires_grad": False,
                "md5": "2e3fa576"
            }
        }
        debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
        debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
        assert deep_compare(debug_json_dict["data"]["data_dict.0.debug"], target_debug_info)

    def test_save_multiple_steps(self):
        data = {"a": torch.Tensor([1., 2.])}
        step = [0, 1, 2]
        async_dump = False
        mode = "tensor"
        dump_path = os.path.join(test_dir, "debug_save")
        config_file_path = os.path.join(test_dir, "config.json")
        self.write_config_json(step, async_dump, mode, dump_path, config_file_path)
        debugger =  PrecisionDebugger(config_file_path)
        for _ in step:
            PrecisionDebugger.save(data, "data_dict", save_backward=False)
            PrecisionDebugger.step()
        # check pt file
        for i in step:
            pt_path = os.path.join(dump_path, f"step{i}", "rank", "dump_tensor_data", "data_dict.0.debug.a.pt")
            assert self.check_real_pt(pt_path, data["a"])
        # check debug json
        target_debug_info = {
            "a": {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [
                2
                ],
                "Max": 2.0,
                "Min": 1.0,
                "Mean": 1.5,
                "Norm": 2.2360680103302,
                "requires_grad": False,
                "data_name": "data_dict.0.debug.a.pt"
            }
        }
        for i in step:
            debug_json_path = os.path.join(dump_path, f"step{i}", "rank", "debug.json")
            debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
            assert deep_compare(debug_json_dict["data"]["data_dict.0.debug"], target_debug_info)

    def test_async_save_tensor(self):
        data = {"a": torch.Tensor([1., 2.])}
        step = []
        async_dump = True
        mode = "tensor"
        dump_path = os.path.join(test_dir, "debug_save")
        config_file_path = os.path.join(test_dir, "config.json")

        self.write_config_json(step, async_dump, mode, dump_path, config_file_path)
        debugger =  PrecisionDebugger(config_file_path)
        PrecisionDebugger.save(data, "data_dict", save_backward=False)
        PrecisionDebugger.step()

        # check pt file
        pt_path = os.path.join(dump_path, "step0", "rank", "dump_tensor_data", "data_dict.0.debug.a.pt")
        assert self.check_real_pt(pt_path, data["a"])

        # check debug json
        target_debug_info = {
            "a": {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [
                2
                ],
                "data_name": "data_dict.0.debug.a.pt",
                "Max": 2.0,
                "Min": 1.0,
                "Mean": 1.5,
                "Norm": 2.2360680103302,
                "requires_grad": False,
            }
        }
        debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
        debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
        assert deep_compare(debug_json_dict["data"]["data_dict.0.debug"], target_debug_info)

    def test_save_multiple_times(self):
        data = {"a": torch.Tensor([1., 2.])}
        step = []
        call_times = 3
        async_dump = False
        mode = "tensor"
        dump_path = os.path.join(test_dir, "debug_save")
        config_file_path = os.path.join(test_dir, "config.json")

        self.write_config_json(step, async_dump, mode, dump_path, config_file_path)
        debugger =  PrecisionDebugger(config_file_path)
        for _ in range(call_times):
            PrecisionDebugger.save(data, "data_dict", save_backward=False)
        PrecisionDebugger.step()

        # check pt file
        for i in range(call_times):
            pt_path = os.path.join(dump_path, "step0", "rank", "dump_tensor_data", f"data_dict.{i}.debug.a.pt")
            assert self.check_real_pt(pt_path, data["a"])

        # check debug json
        for i in range(call_times):
            target_debug_info = {
                "a": {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": [
                    2
                    ],
                    "Max": 2.0,
                    "Min": 1.0,
                    "Mean": 1.5,
                    "Norm": 2.2360680103302,
                    "requires_grad": False,
                    "data_name": f"data_dict.{i}.debug.a.pt"
                }
            }

            debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
            debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
            assert deep_compare(debug_json_dict["data"][f"data_dict.{i}.debug"], target_debug_info)

    def test_save_backward(self):
        x = torch.Tensor([1., 2.])
        target_x_grad = torch.Tensor([1., 1.])
        def _forward_simple_func(x):
            PrecisionDebugger.save(x, "x_tensor")
            return x.sum()
        step = []
        async_dump = False
        mode = "tensor"
        dump_path = os.path.join(test_dir, "debug_save")
        config_file_path = os.path.join(test_dir, "config.json")
        self.write_config_json(step, async_dump, mode, dump_path, config_file_path)
        debugger =  PrecisionDebugger(config_file_path)
        x.requires_grad = True
        loss = _forward_simple_func(x)
        loss.backward()
        PrecisionDebugger.step()
        x_info_list = [
            x,
            os.path.join(dump_path, "step0", "rank", "dump_tensor_data", "x_tensor.0.debug.pt"),
            "x_tensor.0.debug",
            {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [
                    2
                ],
                "Max": 2.0,
                "Min": 1.0,
                "Mean": 1.5,
                "Norm": 2.2360680103302,
                "requires_grad": True,
                "data_name": "x_tensor.0.debug.pt"
            },
        ]
        x_grad_info_list = [
            target_x_grad,
            os.path.join(dump_path, "step0", "rank", "dump_tensor_data", "x_tensor_grad.0.debug.pt"),
            "x_tensor_grad.0.debug",
            {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [
                    2
                ],
                "Max": 1.0,
                "Min": 1.0,
                "Mean": 1.0,
                "Norm": 1.4142135381698608,
                "requires_grad": False,
                "data_name": "x_tensor_grad.0.debug.pt"
            },
        ]
        check_list = [x_info_list, x_grad_info_list]
        debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
        debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
        for check_info in check_list:
            target_tensor, target_tensor_path, target_tensor_key, target_tensor_info = check_info
            assert self.check_real_pt(target_tensor_path, target_tensor)
            assert deep_compare(debug_json_dict["data"][target_tensor_key], target_tensor_info)

    def test_save_compilcated_data_structure_backward(self):
        x = torch.Tensor([1., 2.])
        target_x_grad = torch.Tensor([1., 1.])
        def _forward_complicated_func(x):
            complicated_structure = [{"a_key": x}]
            PrecisionDebugger.save(complicated_structure, "complicated_structure")
            return complicated_structure[0]["a_key"].sum()
        step = []
        async_dump = False
        mode = "tensor"
        dump_path = os.path.join(test_dir, "debug_save")
        config_file_path = os.path.join(test_dir, "config.json")
        self.write_config_json(step, async_dump, mode, dump_path, config_file_path)
        debugger =  PrecisionDebugger(config_file_path)
        x.requires_grad = True
        loss = _forward_complicated_func(x)
        loss.backward()
        PrecisionDebugger.step()
        complicated_structure_info_list = [
            x,
            os.path.join(dump_path, "step0", "rank", "dump_tensor_data", "complicated_structure.0.debug.0.a_key.pt"),
            "complicated_structure.0.debug",
            [
                {
                    "a_key": {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": [
                    2
                    ],
                    "Max": 2.0,
                    "Min": 1.0,
                    "Mean": 1.5,
                    "Norm": 2.2360680103302,
                    "requires_grad": True,
                    "data_name": "complicated_structure.0.debug.0.a_key.pt"
                    }
                }
            ],
        ]
        complicated_structure_grad_info_list = [
            target_x_grad,
            os.path.join(dump_path, "step0", "rank", "dump_tensor_data", "complicated_structure_grad.0.debug.0.a_key.pt"),
            "complicated_structure_grad.0.debug",
            [
                {
                    "a_key": {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": [
                    2
                    ],
                    "Max": 1.0,
                    "Min": 1.0,
                    "Mean": 1.0,
                    "Norm": 1.4142135381698608,
                    "requires_grad": False,
                    "data_name": "complicated_structure_grad.0.debug.0.a_key.pt"
                    }
                }
            ],
        ]
        check_list = [complicated_structure_info_list, complicated_structure_grad_info_list]
        debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
        debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
        for check_info in check_list:
            target_tensor, target_tensor_path, target_tensor_key, target_tensor_info = check_info
            assert self.check_real_pt(target_tensor_path, target_tensor)
            assert deep_compare(debug_json_dict["data"][target_tensor_key], target_tensor_info)