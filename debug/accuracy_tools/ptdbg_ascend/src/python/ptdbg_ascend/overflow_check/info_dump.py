import inspect
import fcntl
import os
import threading

import json
import numpy as np
import torch

from ..common.utils import print_error_log, get_time
from ..common.file_check_util import FileOpen


special_torch_object = ["memory_format"]
lock = threading.Lock()


def write_npy(file_path, tensor):
    saved_tensor = tensor.contiguous().cpu().detach()
    if tensor.dtype == torch.bfloat16:
        saved_numpy = saved_tensor.to(torch.float32).numpy()
    else:
        saved_numpy = saved_tensor.numpy()
    if os.path.exists(file_path):
        raise ValueError(f"File {file_path} already exists")
    np.save(file_path, saved_numpy)
    full_path = os.path.abspath(file_path)
    return full_path


class APIInfo:
    def __init__(self, api_name, is_forward, save_real_data=False):
        self.rank = os.getpid()
        self.api_name = api_name
        self.save_real_data = save_real_data
        self.torch_object_key = {'device': self.analyze_device_in_kwargs, 'dtype': self.analyze_dtype_in_kwargs}
        self.is_forward = is_forward
        self.args_num = 0

    def analyze_element(self, element):
        if isinstance(element, (list, tuple)):
            out = []
            for item in element:
                out.append(self.analyze_element(item))
            return out
        elif isinstance(element, dict):
            out_dict = {}
            for key, value in element.items():
                if key in self.torch_object_key.keys():
                    fun = self.torch_object_key[key]
                    out_dict[key] = fun(value)
                elif key in special_torch_object:
                    continue
                else:
                    out_dict[key] = self.analyze_element(value)
            return out_dict
        elif isinstance(element, torch.Tensor):
            out_tensor = self.analyze_tensor(element, self.save_real_data)
            return out_tensor
        elif self.is_builtin_class(element):
            out_builtin = self.analyze_builtin(element)
            return out_builtin
        else:
            msg = f"Type {type(element)} is unsupported at analyze_element"
            print_error_log(msg)

            raise NotImplementedError(msg)

    def analyze_tensor(self, arg, save_real_data):
        single_arg = {}
        if not save_real_data:
            single_arg.update({'type': 'torch.Tensor'})
            single_arg.update({'dtype': str(arg.dtype)})
            single_arg.update({'shape': arg.shape})
            single_arg.update({'Max': self.transfer_types(self.get_tensor_extremum(arg, 'max'), str(arg.dtype))})
            single_arg.update({'Min': self.transfer_types(self.get_tensor_extremum(arg, 'min'), str(arg.dtype))})
            single_arg.update({'requires_grad': arg.requires_grad})

        else:
            dump_path = "./"
            api_args = self.api_name + '.' + str(self.args_num)
            rank = arg.device.index
            if self.is_forward:
                forward_real_data_path = os.path.join(dump_path, "forward_real_data_" + get_time(), f"rank{rank}")
                if not os.path.exists(forward_real_data_path):
                    os.makedirs(forward_real_data_path, 0o755)

                file_path = os.path.join(forward_real_data_path, f'{api_args}.npy')
            else:
                backward_real_data_path = os.path.join(dump_path, "backward_real_data_" + get_time(), f"rank{rank}")
                if not os.path.exists(backward_real_data_path):
                    os.makedirs(backward_real_data_path, 0o755)
                file_path = os.path.join(backward_real_data_path, f'{api_args}.npy')
            self.args_num += 1
            npy_path = write_npy(file_path, arg)
            single_arg.update({'type': 'torch.Tensor'})
            single_arg.update({'datapath': npy_path})
            single_arg.update({'requires_grad': arg.requires_grad})
        return single_arg

    def analyze_builtin(self, arg):
        single_arg = {}
        if isinstance(arg, slice):
            single_arg.update({'type': "slice"})
            single_arg.update({'value': [arg.start, arg.stop, arg.step]})
        else:
            single_arg.update({'type': self.get_type_name(str(type(arg)))})
            single_arg.update({'value': arg})
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
        single_arg.update({'type': 'torch.device'})
        if not isinstance(element, str):

            if hasattr(element, "index"):
                device_value = element.type + ":" + str(element.index)
                single_arg.update({'value': device_value})
            else:
                device_value = element.type
        else:
            single_arg.update({'value': element})
        return single_arg

    def analyze_dtype_in_kwargs(self, element):
        single_arg = {}
        single_arg.update({'type': 'torch.dtype'})
        single_arg.update({'value': str(element)})
        return single_arg

    def get_tensor_extremum(self, data, operator):
        if data.dtype is torch.bool:
            if operator == 'max':
                return True in data
            elif operator == 'min':
                return False not in data
        if operator == 'max':
            return torch._C._VariableFunctionsClass.max(data).item()
        else:
            return torch._C._VariableFunctionsClass.min(data).item()

    def get_type_name(self, name):

        left = name.index("'")
        right = name.rindex("'")
        return name[left + 1: right]


class ForwardAPIInfo(APIInfo):
    def __init__(self, name, save_real_data, args, kwargs):
        super().__init__(name, is_forward=True, save_real_data=save_real_data)
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
        super().__init__(name, is_forward=False)
        self.analyze_api_input(grads)

    def analyze_api_input(self, grads):
        grads_info_list = self.analyze_element(grads)
        self.grad_info_struct = {self.api_name: grads_info_list}


def write_api_info_json(api_info):
    dump_path = "./"
    rank = api_info.rank
    if isinstance(api_info, ForwardAPIInfo):
        file_path = os.path.join(dump_path, f'forward_info_{rank}.json')
        stack_file_path = os.path.join(dump_path, f'stack_info_{rank}.json')
        write_json(file_path, api_info.api_info_struct)
        write_json(stack_file_path, api_info.stack_info_struct, indent=4)

    elif isinstance(api_info, BackwardAPIInfo):
        file_path = os.path.join(dump_path, f'backward_info_{rank}.json')
        write_json(file_path, api_info.grad_info_struct)
    else:
        raise ValueError(f"Invalid api_info type {type(api_info)}")


def write_json(file_path, data, indent=None):
    if not os.path.exists(file_path):
        with FileOpen(file_path, 'w') as f:
            f.write("{\n}")
    lock.acquire()
    with FileOpen(file_path, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.truncate()
            if f.tell() > 3:
                f.seek(f.tell() - 1, os.SEEK_SET)
                f.truncate()
                f.write(',\n')
            f.write(json.dumps(data, indent=indent)[1:-1] + '\n}')
        except Exception as e:
            raise ValueError(f"Json save failed:{e}") from e
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            lock.release()


def initialize_output_json():
    dump_path = os.path.realpath("./")
    files = ['forward_info.json', 'backward_info.json', 'stack_info.json']

    forward_real_data_path = os.path.join(dump_path, 'forward_real_data')
    if os.path.exists(forward_real_data_path):
        raise ValueError(f"file {forward_real_data_path} already exists, please remove it first")
    else:
        os.mkdir(forward_real_data_path, mode=0o750)

    backward_real_data_path = os.path.join(dump_path, 'backward_real_data')
    if os.path.exists(backward_real_data_path):
        raise ValueError(f"file {backward_real_data_path} already exists, please remove it first")
    else:
        os.mkdir(backward_real_data_path, mode=0o750)
    for file in files:
        file_path = os.path.join(dump_path, file)
        if os.path.exists(file_path):
            raise ValueError(f"file {file_path} already exists, please remove it first or use a new dump path")