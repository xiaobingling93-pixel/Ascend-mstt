import os
import yaml
from api_accuracy_checker.common.utils import check_file_or_directory_path
from api_accuracy_checker.hook_module.utils import WrapFunctionalOps, WrapTensorOps, WrapTorchOps
from ptdbg_ascend.src.python.ptdbg_ascend.common.file_check_util import FileOpen

WrapApi = set(WrapFunctionalOps) | set(WrapTensorOps) | set(WrapTorchOps)


class Config:
    def __init__(self, yaml_file):
        check_file_or_directory_path(yaml_file, False)
        with FileOpen(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        self.config = {key: self.validate(key, value) for key, value in config.items()}

    def validate(self, key, value):
        validators = {
            'dump_path': str,
            'jit_compile': bool,
            'real_data': bool,
            'dump_step': int,
            'error_data_path': str,
            'target_iter': list,
            'precision': int,
            'white_list': list,
            'enable_dataloader': bool
        }
        if not isinstance(value, validators.get(key)):
            raise ValueError(f"{key} must be {validators[key].__name__} type")
        if key == 'target_iter':
            if not isinstance(value, list):
                raise ValueError("target_iter must be a list type")
            if any(isinstance(i, bool) for i in value):
                raise ValueError("target_iter cannot contain boolean values")
            if not all(isinstance(i, int) for i in value):
                raise ValueError("All elements in target_iter must be of int type")
            if any(i < 0 for i in value):
                raise ValueError("All elements in target_iter must be greater than or equal to 0")
        if key == 'precision' and value < 0:
            raise ValueError("precision must be greater than 0")
        if key == 'white_list':
            if not isinstance(value, list):
                raise ValueError("white_list must be a list type")
            if not all(isinstance(i, str) for i in value):
                raise ValueError("All elements in white_list must be of str type")
            invalid_api = [i for i in value if i not in WrapApi]
            if invalid_api:
                raise ValueError(f"{', '.join(invalid_api)} is not in support_wrap_ops.yaml, please check the white_list")
        return value

    def __getattr__(self, item):
        return self.config[item]

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    def update_config(self, dump_path=None, real_data=False, target_iter=None, white_list=None):
        args = {
            "dump_path": dump_path if dump_path else self.config.get("dump_path", './'),
            "real_data": real_data,
            "target_iter": target_iter if target_iter else self.config.get("target_iter", [1]),
            "white_list": white_list if white_list else self.config.get("white_list", [])
        }
        for key, value in args.items():
            if key in self.config:
                self.config[key] = self.validate(key, value)
            else:
                raise ValueError(f"Invalid key '{key}'")


cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)