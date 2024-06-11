import os
import hashlib
from abc import ABC, abstractmethod
import torch
from grad_tool.common.utils import print_info_log


class LevelOps:
    @staticmethod
    def intervals_header(bounds):
        intervals = []
        for i, _ in enumerate(bounds):
            if i == 0:
                intervals.append(f"(-inf, {bounds[i]}]")
            else:
                intervals.append(f"({bounds[i-1]}, {bounds[i]}]")
        intervals.extend([f"({bounds[-1]}, inf)", "=0"])
        return intervals 

    @staticmethod
    def count_grad_distribution(grad, bounds):
        grad = grad.cpu().detach()
        if grad.dtype == torch.bfloat16:
            grad = grad.to(torch.float32)
        element_num = grad.numel()
        grad_equal_0_num = (grad == 0).sum().item()
        bound = torch.Tensor(bounds)
        bucketsize_result = torch.bucketize(grad, bound)
        interval_nums = [(bucketsize_result == i).sum().item() for i in range(len(bound) + 1)]
        interval_nums.append(grad_equal_0_num)
        return_list = [x / element_num if element_num != 0 else 0 for x in interval_nums]
        return return_list
    
    @staticmethod
    def save_grad_direction(param_name, grad, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        param_grad = grad.clone().detach()
        is_positive = param_grad > 0
        torch.save(is_positive, f'{save_path}/{param_name}.pt')

    @staticmethod
    def MD5_content(grad):
        tensor_bytes = grad.cpu().detach().float().numpy().tobytes()
        md5_hash = hashlib.md5(tensor_bytes)
        return [md5_hash.hexdigest()]
    
    @staticmethod
    def MD5_header():
        return ["MD5"]
    

class Level(ABC):
    @abstractmethod
    def save_grad_direction(self, param_name, grad, save_path):
        pass

    @abstractmethod
    def count_grad_distribution(self, grad, bounds) -> list:
        pass

    @abstractmethod
    def intervals_header(self, bounds) -> list:
        pass

    @abstractmethod
    def MD5_content(self, grad) -> list:
        pass

    @abstractmethod
    def MD5_header(self) -> list:
        pass


class Level_0(Level):
    def save_grad_direction(self, param_name, grad, save_path):
        pass

    def count_grad_distribution(self, grad, bounds):
        return []

    def intervals_header(self, bounds):
        return []
    
    def MD5_content(self, grad):
        return LevelOps.MD5_content(grad)

    def MD5_header(self):
        return LevelOps.MD5_header()


class Level_1(Level):
    def save_grad_direction(self, param_name, grad, save_path):
        LevelOps.save_grad_direction(param_name, grad, save_path)

    def count_grad_distribution(self, grad, bounds):
        return []

    def intervals_header(self, bounds):
        return []
    
    def MD5_content(self, grad):
        return []

    def MD5_header(self):
        return []


class Level_2(Level):
    def save_grad_direction(self, param_name, grad, save_path):
        LevelOps.save_grad_direction(param_name, grad, save_path)

    def count_grad_distribution(self, grad, bounds):
        return LevelOps.count_grad_distribution(grad, bounds)

    def intervals_header(self, bounds):
        return LevelOps.intervals_header(bounds)
    
    def MD5_content(self, grad):
        return []

    def MD5_header(self):
        return []


class LevelAdapter:
    levels = {"L0": Level_0, "L1": Level_1, "L2": Level_2}

    @staticmethod
    def level_adapter(level):
        if level not in LevelAdapter.levels:
            raise Exception(f"level is valid, not in {LevelAdapter.levels.keys()}")
        return LevelAdapter.levels[level]()
