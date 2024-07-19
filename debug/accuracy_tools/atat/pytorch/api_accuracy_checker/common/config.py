import os
import yaml
from atat.pytorch.api_accuracy_checker.common.utils import check_file_or_directory_path
from atat.pytorch.hook_module.utils import WrapFunctionalOps, WrapTensorOps, WrapTorchOps
from atat.core.common.file_check import FileOpen

WrapApi = set(WrapFunctionalOps) | set(WrapTensorOps) | set(WrapTorchOps)


class Config:
    def __init__(self, yaml_file):
        check_file_or_directory_path(yaml_file, False)
        with FileOpen(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
        self.config = {key: self.validate(key, value) for key, value in config.items()}

    def __getattr__(self, item):
        return self.config[item]

    def __str__(self):
        return '\n'.join(f"{key}={value}" for key, value in self.config.items())

    @staticmethod
    def validate(key, value):
        validators = {
            'white_list': list,
            'error_data_path': str,
            'precision': int
        }
        if key not in validators:
            raise ValueError(f"{key} must be one of {validators.keys()}")
        if not isinstance(value, validators.get(key)):
            raise ValueError(f"{key} must be {validators[key].__name__} type")
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


cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
yaml_path = os.path.join(cur_path, "config.yaml")
msCheckerConfig = Config(yaml_path)
