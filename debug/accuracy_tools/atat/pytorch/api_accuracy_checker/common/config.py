import os
import yaml
from ..common.utils import check_file_or_directory_path
from ...hook_module.utils import WrapFunctionalOps, WrapTensorOps, WrapTorchOps
from ...common.file_check import FileOpen

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
            'real_data': bool,
            'enable_dataloader': bool,
            'target_iter': list,
            'white_list': list,
            'error_data_path': str,
            'jit_compile': bool,
            'precision': int
        }
        if key not in validators:
            raise ValueError(f"{key} must be one of {validators.keys()}")
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
                raise ValueError(
                    f"{', '.join(invalid_api)} is not in support_wrap_ops.yaml, please check the white_list")
        return value

    def __getattr__(self, item):
        return self.config[item]

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    def update_config(self, dump_path=None, real_data=None, target_iter=None, white_list=None, enable_dataloader=None):
        args = {
            "dump_path": dump_path if dump_path else self.config.get("dump_path", './'),
            "real_data": real_data if real_data else self.config.get("real_data", False),
            "target_iter": target_iter if target_iter else self.config.get("target_iter", [1]),
            "white_list": white_list if white_list else self.config.get("white_list", []),
            "enable_dataloader": enable_dataloader if enable_dataloader else self.config.get("enable_dataloader", False)
        }
        for key, value in args.items():
            if key in self.config:
                self.config[key] = self.validate(key, value)
            else:
                raise ValueError(f"Invalid key '{key}'")


cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)
