import os
import torch
from api_accuracy_checker.common.utils import print_error_log, write_pt, create_directory
from ptdbg_ascend.src.python.ptdbg_ascend.common.utils import check_path_before_create
from api_accuracy_checker.common.config import msCheckerConfig


class BaseAPIInfo:
    def __init__(self, api_name, is_forward, is_save_data, save_path, forward_path, backward_path):
        self.rank = os.getpid()
        self.api_name = api_name
        self.torch_object_key = {'device': self.analyze_device_in_kwargs, 'dtype': self.analyze_dtype_in_kwargs}
        self.is_forward = is_forward
        self.args_num = 0
        self.is_save_data = is_save_data
        self.save_path = save_path
        self.forward_path = forward_path
        self.backward_path = backward_path

    def analyze_element(self, element):
        if isinstance(element, (list, tuple)):
            out = []
            for item in element:
                out.append(self.analyze_element(item))
        elif isinstance(element, dict):
            out = {}
            for key, value in element.items():
                if key in self.torch_object_key.keys():
                    fun = self.torch_object_key[key]
                    out[key] = fun(value)
                else:
                    out[key] = self.analyze_element(value)

        elif isinstance(element, torch.Tensor):
            out = self.analyze_tensor(element)

        elif self.is_builtin_class(element):
            out = self.analyze_builtin(element)
        else:
            msg = f"Type {type(element)} is unsupported at analyze_element"
            print_error_log(msg)

            raise NotImplementedError(msg)
        return out

    def analyze_tensor(self, arg):
        single_arg = {}
        if not self.is_save_data:

            single_arg.update({'type' : 'torch.Tensor'})
            single_arg.update({'dtype' : str(arg.dtype)})
            single_arg.update({'shape' : arg.shape})
            single_arg.update({'Max' : self.transfer_types(self.get_tensor_extremum(arg, 'max'), str(arg.dtype))})
            single_arg.update({'Min' : self.transfer_types(self.get_tensor_extremum(arg, 'min'), str(arg.dtype))})
            single_arg.update({'requires_grad': arg.requires_grad})

        else:
            api_args = self.api_name + '.' + str(self.args_num)
            from api_accuracy_checker.dump.dump import DumpUtil
            if self.is_forward:
                forward_real_data_path = os.path.join(self.save_path, "step" + str((DumpUtil.call_num - 1) if msCheckerConfig.enable_dataloader else DumpUtil.call_num), self.forward_path, "rank" + str(self.rank))
                check_path_before_create(forward_real_data_path)
                create_directory(forward_real_data_path)
                file_path = os.path.join(forward_real_data_path, f'{api_args}.pt')
            else:
                backward_real_data_path = os.path.join(self.save_path, "step" + str((DumpUtil.call_num - 1) if msCheckerConfig.enable_dataloader else DumpUtil.call_num), self.backward_path, "rank" + str(self.rank))
                check_path_before_create(backward_real_data_path)
                create_directory(backward_real_data_path)
                file_path = os.path.join(backward_real_data_path, f'{api_args}.pt')
            self.args_num += 1
            pt_path = write_pt(file_path, arg.contiguous().cpu().detach())
            single_arg.update({'type' : 'torch.Tensor'})
            single_arg.update({'datapath' : pt_path})
            single_arg.update({'requires_grad': arg.requires_grad})
        return single_arg

    def analyze_builtin(self, arg):
        single_arg = {}
        if self.is_save_data:
            self.args_num += 1
        if isinstance(arg, slice):
            single_arg.update({'type' : "slice"})
            single_arg.update({'value' : [arg.start, arg.stop, arg.step]})
        else:
            single_arg.update({'type' : self.get_type_name(str(type(arg)))})
            single_arg.update({'value' : arg})
        return single_arg

    def transfer_types(self, data, dtype):
        if 'int' in dtype or 'bool' in dtype:
            return int(data)
        else:
            return float(data)

    def is_builtin_class(self, element):
        if element is None or isinstance(element, (bool, int, float, str, slice)):
            return True
        return False

    def analyze_device_in_kwargs(self, element):
        single_arg = {}
        single_arg.update({'type' : 'torch.device'})
        if not isinstance(element, str):

            if hasattr(element, "index"):
                device_value = element.type + ":" + str(element.index)
                single_arg.update({'value' : device_value})
            else:
                device_value = element.type
        else:
            single_arg.update({'value' : element})
        return single_arg

    def analyze_dtype_in_kwargs(self, element):
        single_arg = {}
        single_arg.update({'type' : 'torch.dtype'})
        single_arg.update({'value' : str(element)})
        return single_arg

    def get_tensor_extremum(self, data, operator):
        if data.dtype is torch.bool:
            if operator == 'max':
                return True in data
            elif operator == 'min':
                return False not in data
        if operator == 'max':
            return torch._C._VariableFunctionsClass.max(data.float()).item()
        else:
            return torch._C._VariableFunctionsClass.min(data.float()).item()

    def get_type_name(self, name):

        left = name.index("'")
        right = name.rindex("'")
        return name[left + 1 : right]
