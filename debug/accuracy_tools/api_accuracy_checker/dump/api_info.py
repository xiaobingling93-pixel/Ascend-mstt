# 定义API INFO，保存基本信息，用于后续结构体的落盘，注意考虑random场景及真实数据场景
import os
import inspect
import torch
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.common.base_api import BaseAPIInfo


class APIInfo(BaseAPIInfo):
    def __init__(self, api_name, save_path, save_dir_name):
        super().__init__(api_name, save_path, save_dir_name)
        self.torch_object_key = {'device': self.analyze_device_in_kwargs, 'dtype': self.analyze_dtype_in_kwargs}
        self.rank = os.getpid()
        self.save_real_data = msCheckerConfig.real_data
        self.save_path = save_path
        self.save_dir_name = save_dir_name

    @staticmethod
    def analyze_device_in_kwargs(element):
        single_arg = {}
        single_arg.update({'type': 'torch.device'})
        if not isinstance(element, str):
            if hasattr(element, "index"):
                device_value = element.type + ":" + str(element.index)
            else:
                device_value = element.type
            single_arg.update({'value': device_value})
        else:
            single_arg.update({'value': element})
        return single_arg

    @staticmethod
    def analyze_dtype_in_kwargs(element):
        single_arg = {}
        single_arg.update({'type': 'torch.dtype'})
        single_arg.update({'value': str(element)})
        return single_arg

    @staticmethod
    def get_tensor_extremum(data, operator):
        if data.dtype is torch.bool:
            if operator == 'max':
                return True in data
            elif operator == 'min':
                return False not in data
        if operator == 'max':
            return torch._C._VariableFunctionsClass.max(data.float()).item()
        else:
            return torch._C._VariableFunctionsClass.min(data.float()).item()

    @staticmethod
    def get_type_name(name):
        left = name.index("'")
        right = name.rindex("'")
        return name[left + 1: right]

    @staticmethod
    def transfer_types(data, dtype):
        if 'int' in dtype or 'bool' in dtype:
            return int(data)
        else:
            return float(data)

    def analyze_tensor(self, arg):
        single_arg = {}
        if not self.save_real_data:
            single_arg.update({'type': 'torch.Tensor'})
            single_arg.update({'dtype': str(arg.dtype)})
            single_arg.update({'shape': arg.shape})
            single_arg.update({'Max': self.transfer_types(self.get_tensor_extremum(arg, 'max'), str(arg.dtype))})
            single_arg.update({'Min': self.transfer_types(self.get_tensor_extremum(arg, 'min'), str(arg.dtype))})
            single_arg.update({'requires_grad': arg.requires_grad})
        else:
            from api_accuracy_checker.dump.dump import DumpUtil
            step_dir = "step" + str(DumpUtil.call_num - 1 if msCheckerConfig.enable_dataloader else DumpUtil.call_num)
            rank_dir = f"rank{self.rank}"
            self.full_save_path = os.path.join(self.save_path, step_dir, self.save_dir_name, rank_dir)
            pt_path = super().analyze_tensor(arg)
            single_arg.update({'type': 'torch.Tensor'})
            single_arg.update({'datapath': pt_path})
            single_arg.update({'requires_grad': arg.requires_grad})
        return single_arg

    def analyze_builtin(self, arg):
        single_arg = {}
        if self.save_real_data:
            self.args_num += 1
        if isinstance(arg, slice):
            single_arg.update({'type': "slice"})
            single_arg.update({'value': [arg.start, arg.stop, arg.step]})
        else:
            single_arg.update({'type': self.get_type_name(str(type(arg)))})
            single_arg.update({'value': arg})
        return single_arg


class ForwardAPIInfo(APIInfo):
    def __init__(self, name, args, kwargs):
        super().__init__(name, save_path=msCheckerConfig.dump_path, save_dir_name='forward_real_data')
        self.api_info_struct = {}
        self.stack_info_struct = {}
        self.analyze_api_input(args, kwargs)
        self.analyze_api_call_stack()

    def analyze_api_input(self, args, kwargs):
        args_info_list = self.analyze_element(args)
        kwargs_info_dict = self.analyze_element(kwargs)
        self.api_info_struct = {self.api_name: {"args": args_info_list, "kwargs": kwargs_info_dict}}

    def analyze_api_call_stack(self):
        stack_str = []
        for (_, path, line, func, code, _) in inspect.stack()[3:]:
            if not code:
                continue
            stack_line = " ".join([
                "File", ", ".join([path, " ".join(["line", str(line)]), " ".join(["in", func]),
                                   " ".join(["\n", code[0].strip()])])])
            stack_str.append(stack_line)
        self.stack_info_struct = {self.api_name: stack_str}


class BackwardAPIInfo(APIInfo):
    def __init__(self, name, grads):
        super().__init__(name, save_path=msCheckerConfig.dump_path, save_dir_name='backward_real_data')
        self.grad_info_struct = {}
        self.analyze_api_input(grads)

    def analyze_api_input(self, grads):
        grads_info_list = self.analyze_element(grads)
        self.grad_info_struct = {self.api_name: grads_info_list}
