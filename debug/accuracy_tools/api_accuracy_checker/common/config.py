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
            'real_data': bool,
            'enable_dataloader': bool,
            'target_iter': list,
            'white_list': list,
            'error_data_path': str,
            'precision': int,
            'is_online': bool,
            'nfs_path': str,
            'is_benchmark_device': bool,
            'host': str,
            'port': int,
            'rank_list': list
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
                raise ValueError(f"{', '.join(invalid_api)} is not in support_wrap_ops.yaml, please check the white_list")
        if key == 'nfs_path':
            if value and not os.path.exists(value):
                raise ValueError(f"nfs path {value} doesn't exist.")
        return value

    def __getattr__(self, item):
        return self.config[item]

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    def update_config(self, dump_path=None, real_data=None, target_iter=None, white_list=None, enable_dataloader=None,
                      is_online=None, is_benchmark_device=True, port=None, host=None, rank_list=None):
        args = {
            "dump_path": dump_path if dump_path is not None else self.config.get("dump_path", './'),
            "real_data": real_data if real_data is not None else self.config.get("real_data", False),
            "target_iter": target_iter if target_iter is not None else self.config.get("target_iter", [1]),
            "white_list": white_list if white_list is not None else self.config.get("white_list", []),
            "enable_dataloader": enable_dataloader
            if enable_dataloader is not None else self.config.get("enable_dataloader", False),
            "is_online": is_online if is_online is not None else self.config.get("is_online", False),
            "is_benchmark_device": is_benchmark_device,
            "host": host if host is not None else self.config.get("host", ""),
            "port": port if port is not None else self.config.get("port", -1),
            "rank_list": rank_list if rank_list is not None else self.config.get("rank_list", [0])
        }
        for key, value in args.items():
            if key in self.config:
                self.config[key] = self.validate(key, value)
            else:
                raise ValueError(f"Invalid key '{key}'")


cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)