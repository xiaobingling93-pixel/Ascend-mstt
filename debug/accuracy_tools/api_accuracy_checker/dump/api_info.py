# 定义API INFO，保存基本信息，用于后续结构体的落盘，注意考虑random场景及真实数据场景
import os
import inspect
import torch
import numpy as np
from api_accuracy_checker.common.config import msCheckerConfig
from api_accuracy_checker.common.utils import print_error_log, write_pt, create_directory, DumpException, \
    get_real_data_path
from ptdbg_ascend.src.python.ptdbg_ascend.common.utils import check_path_before_create


def get_tensor_extremum(data, operator):
    if data.dtype is torch.bool:
        if data.numel() == 0:
            return False, False
        if operator == 'max':
            return True in data, True in data
        elif operator == 'min':
            return False not in data, False not in data
    data_clone = data.float().clone().detach()
    if operator == 'max':
        max_result = torch._C._VariableFunctionsClass.max(data_clone).item()
        if np.isinf(max_result) or np.isnan(max_result):
            return handle_tensor_extremum_nan_inf(data_clone, operator), max_result
        else:
            return max_result, max_result
    else:
        min_result = torch._C._VariableFunctionsClass.min(data_clone).item()
        if np.isinf(min_result) or np.isnan(min_result):
            return handle_tensor_extremum_nan_inf(data_clone, operator), min_result
        else:
            return min_result, min_result


def handle_tensor_extremum_nan_inf(data_clone, operator):
    data_nan = torch._C._VariableFunctionsClass.isnan(data_clone)
    if int(torch._C._VariableFunctionsClass.sum(data_nan)) == data_clone.numel():
        return float('nan')
    finite_mask = torch._C._VariableFunctionsClass.isfinite(data_clone)
    if int(torch._C._VariableFunctionsClass.sum(finite_mask)) > 0:
        finite_values = data_clone[finite_mask]
        return torch._C._VariableFunctionsClass.max(finite_values).item() if operator == 'max' else \
         torch._C._VariableFunctionsClass.min(finite_values).item()
    else:
        data_no_nan = data_clone[~data_nan]
        return torch._C._VariableFunctionsClass.max(data_no_nan).item() if operator == 'max' else \
         torch._C._VariableFunctionsClass.min(data_no_nan).item()


def get_type_name(name):
    left = name.index("'")
    right = name.rindex("'")
    return name[left + 1: right]


def transfer_types(data, dtype):
    if 'int' in dtype or 'bool' in dtype:
        return int(data)
    else:
        return float(data)


def is_builtin_class(element):
    return element is None or isinstance(element, (bool, int, float, str, slice))


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


def analyze_dtype_in_kwargs(element):
    single_arg = {}
    single_arg.update({'type': 'torch.dtype'})
    single_arg.update({'value': str(element)})
    return single_arg


class APIInfo:
    def __init__(self, api_name, save_path, is_save_data=False):
        self.api_name = api_name
        self.torch_object_key = {'device': analyze_device_in_kwargs, 'dtype': analyze_dtype_in_kwargs}
        self.rank = os.getpid()
        self.is_save_data = is_save_data
        self.save_path = save_path
        self.args_num = 0

    @staticmethod
    def get_full_save_path(save_path, dir_name, contain_step=False):
        if contain_step:
            from api_accuracy_checker.dump.dump import DumpUtil
            step_dir = "step" + str(DumpUtil.call_num - 1 if msCheckerConfig.enable_dataloader else DumpUtil.call_num)
            rank_dir = f"rank{os.getpid()}"
            return os.path.join(save_path, step_dir, dir_name, rank_dir)
        else:
            return os.path.join(save_path, dir_name)

    def analyze_element(self, element):
        if isinstance(element, (list, tuple)):
            out = []
            for item in element:
                out.append(self.analyze_element(item))
            return out

        if isinstance(element, dict):
            out_dict = {}
            for key, value in element.items():
                if key in self.torch_object_key.keys():
                    fun = self.torch_object_key[key]
                    out_dict[key] = fun(value)
                else:
                    out_dict[key] = self.analyze_element(value)
            return out_dict
        
        converted_numpy, numpy_type = self._convert_numpy_to_builtin(element)
        if converted_numpy is not element:
            return self._analyze_numpy(converted_numpy, numpy_type)

        if isinstance(element, torch.Tensor):
            return self._analyze_tensor(element)

        if is_builtin_class(element):
            return self._analyze_builtin(element)

        msg = f"Type {type(element)} is unsupported at analyze_element"
        print_error_log(msg)
        raise DumpException(DumpException.INVALID_DATA_ERROR)

    def _analyze_tensor(self, arg):
        single_arg = {}
        if not self.is_save_data:
            single_arg.update({'type': 'torch.Tensor'})
            single_arg.update({'dtype': str(arg.dtype)})
            single_arg.update({'shape': arg.shape})
            max_handle, max_origin = get_tensor_extremum(arg, 'max')
            single_arg.update({'Max': transfer_types(max_handle, str(arg.dtype))})
            single_arg.update({'Max_origin': transfer_types(max_origin, str(arg.dtype))})
            min_handle, min_origin = get_tensor_extremum(arg, 'min')
            single_arg.update({'Min': transfer_types(min_handle, str(arg.dtype))})
            single_arg.update({'Min_origin': transfer_types(min_origin, str(arg.dtype))})
            single_arg.update({'requires_grad': arg.requires_grad})
        else:
            api_args = self.api_name + '.' + str(self.args_num)
            check_path_before_create(self.save_path)
            create_directory(self.save_path)
            file_path = os.path.join(self.save_path, f'{api_args}.pt')
            pt_path = write_pt(file_path, arg.contiguous().cpu().detach())
            self.args_num += 1
            real_data_path = get_real_data_path(pt_path)
            single_arg.update({'type': 'torch.Tensor'})
            single_arg.update({'datapath': real_data_path})
            single_arg.update({'requires_grad': arg.requires_grad})
        return single_arg

    def _analyze_builtin(self, arg):
        single_arg = {}
        if self.is_save_data:
            self.args_num += 1
        if isinstance(arg, slice):
            single_arg.update({'type': "slice"})
            single_arg.update({'value': [arg.start, arg.stop, arg.step]})
        else:
            single_arg.update({'type': get_type_name(str(type(arg)))})
            single_arg.update({'value': arg})
        return single_arg
    
    def _analyze_numpy(self, value, numpy_type):
        single_arg = {}
        if self.is_save_data:
            self.args_num += 1
        single_arg.update({'type': numpy_type})
        single_arg.update({'value': value})
        return single_arg
    
    def _convert_numpy_to_builtin(self, arg):
        type_mapping = {
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
            np.complexfloating: complex,
            np.str_: str,
            np.bytes_: bytes,
            np.unicode_: str
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(arg, numpy_type):
                return builtin_type(arg), get_type_name(str(type(arg)))
        return arg, ''


class ForwardAPIInfo(APIInfo):
    def __init__(self, name, args, kwargs):
        super().__init__(name,
                         self.get_full_save_path(msCheckerConfig.dump_path, 'forward_real_data', contain_step=True),
                         is_save_data=msCheckerConfig.real_data)
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
        super().__init__(name,
                         self.get_full_save_path(msCheckerConfig.dump_path, 'backward_real_data', contain_step=True),
                         is_save_data=msCheckerConfig.real_data)
        self.grad_info_struct = {}
        self.analyze_api_input(grads)

    def analyze_api_input(self, grads):
        grads_info_list = self.analyze_element(grads)
        self.grad_info_struct = {self.api_name: grads_info_list}
