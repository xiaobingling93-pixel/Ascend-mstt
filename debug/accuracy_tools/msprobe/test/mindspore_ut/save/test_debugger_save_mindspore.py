import unittest
import os
import json
import mindspore
import numpy as np
import shutil
from unittest.mock import patch

from msprobe.mindspore import PrecisionDebugger
from msprobe.core.data_dump.data_processor.mindspore_processor import MindsporeDataProcessor
from msprobe.mindspore.dump.hook_cell.api_register import get_api_register


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
    def check_real_npy(npy_path, target_ms_tensor, check_values=True, rtol=1e-5, atol=1e-8):
        """
        Enhanced version with optional value comparison.

        Args:
            npy_path (str): Path to the .npy file
            target_ms_tensor: Target mindspore tensor to compare
            check_values (bool): If True, also compare array values
            rtol, atol: Relative and absolute tolerances for value comparison

        Returns:
            bool: True if all checks pass
        """
        # Convert mindspore tensor to numpy if needed
        if hasattr(target_ms_tensor, 'numpy'):
            target_ms_tensor = target_ms_tensor.numpy()
        # Load the npy file
        try:
            npy_data = np.load(npy_path)
        except FileNotFoundError:
            print(f"Error: The file {npy_path} does not exist.")
            return False
        except Exception as e:
            print(f"Error loading npy file: {e}")
            return False
        # Check shapes
        if npy_data.shape != target_ms_tensor.shape:
            print(f"Shape mismatch: npy data shape is {npy_data.shape}, target tensor shape is {target_ms_tensor.shape}")
            return False
        # Check dtypes
        if npy_data.dtype != target_ms_tensor.dtype:
            print(f"Shape mismatch: npy data dtype is {npy_data.dtype}, target tensor dtype is {target_ms_tensor.dtype}")
            return False
        # Optionally check values
        if check_values:
            if not np.allclose(npy_data, target_ms_tensor, rtol=rtol, atol=atol):
                print("Value mismatch: npy data and target tensor values do not match within the specified tolerances.")
                return False

        return True

    def setUp(self):
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        PrecisionDebugger._instance = None
        self.original_mindspore_special_type = MindsporeDataProcessor.mindspore_special_type
        MindsporeDataProcessor.mindspore_special_type = tuple([mindspore.Tensor])

    def tearDown(self):
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        PrecisionDebugger._instance = None
        MindsporeDataProcessor.mindspore_special_type = self.original_mindspore_special_type
        get_api_register(True).restore_all_api()

    @patch("msprobe.mindspore.debugger.precision_debugger.set_register_backward_hook_functions")
    def test_save_real_tensor(self, _):
        data = {"a": mindspore.Tensor([1., 2.])}
        step = []
        async_dump = False
        mode = "tensor"
        dump_path = os.path.join(test_dir, "debug_save")
        config_file_path = os.path.join(test_dir, "config.json")

        self.write_config_json(step, async_dump, mode, dump_path, config_file_path)
        debugger =  PrecisionDebugger(config_file_path)
        PrecisionDebugger.save(data, "data_dict", save_backward=False)
        PrecisionDebugger.step()

        # check npy file
        npy_path = os.path.join(dump_path, "step0", "rank", "dump_tensor_data", "data_dict.0.debug.a.npy")
        assert self.check_real_npy(npy_path, data["a"])

        # check debug json
        target_debug_info = {
            "a": {
                "type": "mindspore.Tensor",
                "dtype": "Float32",
                "shape": [
                2
                ],
                "Max": 2.0,
                "Min": 1.0,
                "Mean": 1.5,
                "Norm": 2.2360680103302,
                "data_name": "data_dict.0.debug.a.npy"
            }
        }
        debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
        debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
        assert deep_compare(debug_json_dict["data"]["data_dict.0.debug"], target_debug_info)

    @patch("msprobe.mindspore.debugger.precision_debugger.set_register_backward_hook_functions")
    def test_save_md5(self, _):
        data = {"a": mindspore.Tensor([1., 2.])}
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
                "type": "mindspore.Tensor",
                "dtype": "Float32",
                "shape": [
                2
                ],
                "Max": 2.0,
                "Min": 1.0,
                "Mean": 1.5,
                "Norm": 2.2360680103302,
                "md5": "2e3fa576"
            }
        }
        debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
        debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
        assert deep_compare(debug_json_dict["data"]["data_dict.0.debug"], target_debug_info)

    @patch("msprobe.mindspore.debugger.precision_debugger.set_register_backward_hook_functions")
    def test_save_multiple_steps(self, _):
        data = {"a": mindspore.Tensor([1., 2.])}
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
        # check npy file
        for i in step:
            npy_path = os.path.join(dump_path, f"step{i}", "rank", "dump_tensor_data", "data_dict.0.debug.a.npy")
            assert self.check_real_npy(npy_path, data["a"])
        # check debug json
        target_debug_info = {
            "a": {
                "type": "mindspore.Tensor",
                "dtype": "Float32",
                "shape": [
                2
                ],
                "Max": 2.0,
                "Min": 1.0,
                "Mean": 1.5,
                "Norm": 2.2360680103302,
                "data_name": "data_dict.0.debug.a.npy"
            }
        }
        for i in step:
            debug_json_path = os.path.join(dump_path, f"step{i}", "rank", "debug.json")
            debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
            assert deep_compare(debug_json_dict["data"]["data_dict.0.debug"], target_debug_info)

    @patch("msprobe.mindspore.debugger.precision_debugger.set_register_backward_hook_functions")
    def test_save_multiple_times(self, _):
        data = {"a": mindspore.Tensor([1., 2.])}
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
        # check npy file
        for i in range(call_times):
            npy_path = os.path.join(dump_path, "step0", "rank", "dump_tensor_data", f"data_dict.{i}.debug.a.npy")
            assert self.check_real_npy(npy_path, data["a"])
        # check debug json
        for i in range(call_times):
            target_debug_info = {
                "a": {
                    "type": "mindspore.Tensor",
                    "dtype": "Float32",
                    "shape": [
                    2
                    ],
                    "Max": 2.0,
                    "Min": 1.0,
                    "Mean": 1.5,
                    "Norm": 2.2360680103302,
                    "data_name": f"data_dict.{i}.debug.a.npy"
                }
            }
            debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
            debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
            assert deep_compare(debug_json_dict["data"][f"data_dict.{i}.debug"], target_debug_info)

    @patch("msprobe.mindspore.debugger.precision_debugger.set_register_backward_hook_functions")
    def test_save_compilcated_data_structure(self, _):
        x = mindspore.Tensor([1., 2.])
        complicated_structure = [{"a_key": x}]
        step = []
        async_dump = False
        mode = "tensor"
        dump_path = os.path.join(test_dir, "debug_save")
        config_file_path = os.path.join(test_dir, "config.json")
        self.write_config_json(step, async_dump, mode, dump_path, config_file_path)
        debugger =  PrecisionDebugger(config_file_path)
        PrecisionDebugger.save(complicated_structure, "complicated_structure")
        PrecisionDebugger.step()
        complicated_structure_info_list = [
            x,
            os.path.join(dump_path, "step0", "rank", "dump_tensor_data", "complicated_structure.0.debug.0.a_key.npy"),
            "complicated_structure.0.debug",
            [
                {
                    "a_key": {
                    "type": "mindspore.Tensor",
                    "dtype": "Float32",
                    "shape": [
                    2
                    ],
                    "Max": 2.0,
                    "Min": 1.0,
                    "Mean": 1.5,
                    "Norm": 2.2360680103302,
                    "data_name": "complicated_structure.0.debug.0.a_key.npy"
                    }
                }
            ],
        ]
        debug_json_path = os.path.join(dump_path, "step0", "rank", "debug.json")
        debug_json_dict = self.read_debug_json_into_dict(debug_json_path)
        target_tensor, target_tensor_path, target_tensor_key, target_tensor_info = complicated_structure_info_list
        assert self.check_real_npy(target_tensor_path, target_tensor)
        assert deep_compare(debug_json_dict["data"][target_tensor_key], target_tensor_info)