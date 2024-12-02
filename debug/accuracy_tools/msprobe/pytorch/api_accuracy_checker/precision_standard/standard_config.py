import torch


class StandardConfig:
    
    _small_value = {
        torch.float16: 1e-3,
        torch.bfloat16: 1e-3,
        torch.float32: 1e-6,
        "default": 1e-6
        }
    _small_value_atol = {
        torch.float16: 1e-5,
        torch.bfloat16: 1e-5,
        torch.float32: 1e-9,
        "default": 1e-9
        }
    _rtol = {
        torch.float16: 1e-3,
        torch.bfloat16: 4e-3,
        torch.float32: 1e-6,
        "default": 1e-6  # 默认值也放在配置类中
    }

    @classmethod
    def get_small_valuel(cls, dtype):
        return cls._small_value.get(dtype, cls._small_value["default"])
    
    @classmethod
    def get_small_value_atol(cls, dtype):
        return cls._small_value_atol.get(dtype, cls._small_value_atol["default"])
    
    @classmethod
    def get_rtol(cls, dtype):
        return cls._rtol.get(dtype, cls._rtol["default"])
