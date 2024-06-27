import json
import re
import abc
import torch
from kj600.utils import check_file_before_read


def get_config(file_path='config.json'):
    check_file_before_read(file_path)
    with open(file_path, 'r') as file:
        config = json.load(file)
        return config

# 用于存储所有validator实现类的注册表
config_validator_registry = {}


def register_config_validator(cls):
    """装饰器 用于注册ConfigValidator的实现类"""
    config_validator_registry[cls.__name__] = cls
    return cls


class ConfigValidator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def check_pattern_match(self, config_spec:str):
        pass

    @abc.abstractmethod
    def validate(self, actual_data, module_name:str, data_type:str, pattern_match):
        pass


@register_config_validator
class TensorValidator(ConfigValidator):
    def check_pattern_match(self, config_spec:str):
        pattern = re.compile(r"tensor")
        return pattern.match(config_spec)    

    def validate(self, actual_data, module_name:str, data_type:str, pattern_match):
        if not torch.is_tensor(actual_data):
            raise ValueError(f"Format of {module_name} {data_type} does not match the required format 'tensor' in config.")
        return None


@register_config_validator    
class TupleValidator(ConfigValidator):
    def check_pattern_match(self, config_spec:str):
        pattern = re.compile(r"tuple\[(\d+)\]:(\d+)")
        return pattern.match(config_spec)  
    
    def validate(self, actual_data, module_name: str, data_type: str, pattern_match):
        length, index = map(int, pattern_match.groups())
        if not (0 <= index < length):
            raise ValueError(f"Format of {module_name} {data_type} in config.json does not match the required format 'tuple[x]:y'. y must be greater than or equal to 0 and less than x.")
        if not isinstance(actual_data, tuple):
            raise ValueError(f"Type of {module_name} {data_type} does not match spec of config.json, should be tuple, please check.")
        if len(actual_data) != length:
            raise ValueError(f"Length of {module_name} {data_type} does not match spec of config.json, should be {length}, actual is {len(actual_data)} please check.")
        return index


def validate_config_spec(config_spec:str, actual_data, module_name:str, data_type:str):
    for _, validator_cls in config_validator_registry.items():
        config_validator = validator_cls()
        pattern_match = config_validator.check_pattern_match(config_spec)
        if pattern_match:
            focused_col = config_validator.validate(actual_data, module_name, data_type, pattern_match)
            return focused_col
    raise ValueError(f"config spec in {module_name} {data_type} not supported, expected spec:'tuple\[(\d+)\]:(\d+)' or 'tensor', actual spec: {config_spec}.")
