import os
from abc import ABC, abstractmethod
import torch
from grad_tool.utils import print_info_log


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


class Level_0(Level):
    def save_grad_direction(self, param_name, grad, save_path):
        pass

    def count_grad_distribution(self, grad, bounds):
        return []

    def intervals_header(self, bounds):
        return []


class Level_1(Level):
    def save_grad_direction(self, param_name, grad, save_path):
        pass

    def count_grad_distribution(self, grad, bounds):
        return LevelOps.count_grad_distribution(grad, bounds)

    def intervals_header(self, bounds):
        return LevelOps.intervals_header(bounds)


class Level_2(Level):
    def save_grad_direction(self, param_name, grad, save_path):
        LevelOps.save_grad_direction(param_name, grad, save_path)

    def count_grad_distribution(self, grad, bounds):
        return []

    def intervals_header(self, bounds):
        return []


class Level_3(Level):
    def save_grad_direction(self, param_name, grad, save_path):
        LevelOps.save_grad_direction(param_name, grad, save_path)

    def count_grad_distribution(self, grad, bounds):
        return LevelOps.count_grad_distribution(grad, bounds)

    def intervals_header(self, bounds):
        return LevelOps.intervals_header(bounds)


class LevelAdapter:
    levels = {"L0": Level_0, "L1": Level_1, "L2": Level_2, "L3": Level_3}

    @staticmethod
    def level_adapter(level):
        if level not in LevelAdapter.levels:
            raise Exception(f"level is valid, not in {LevelAdapter.levels.keys()}")
        return LevelAdapter.levels[level]()
