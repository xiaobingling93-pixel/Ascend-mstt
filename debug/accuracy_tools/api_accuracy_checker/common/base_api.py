import os
import torch
from api_accuracy_checker.common.utils import print_error_log, write_pt, create_directory
from ptdbg_ascend.src.python.ptdbg_ascend.common.utils import check_path_before_create


class BaseAPIInfo:
    def __init__(self, api_name, save_path, save_dir_name):
        self.api_name = api_name
        self.torch_object_key = {}
        self.args_num = 0
        self.save_path = save_path
        self.full_save_path = os.path.join(self.save_path, save_dir_name)

    @staticmethod
    def is_builtin_class(element):
        if element is None or isinstance(element, (bool, int, float, str, slice)):
            return True
        return False

    def analyze_element(self, element):
        if isinstance(element, (list, tuple)):
            out = []
            for item in element:
                out.append(self.analyze_element(item))
            return out

        if isinstance(element, dict):
            out = {}
            for key, value in element.items():
                if key in self.torch_object_key.keys():
                    fun = self.torch_object_key[key]
                    out[key] = fun(value)
                else:
                    out[key] = self.analyze_element(value)
            return out

        if isinstance(element, torch.Tensor):
            return self.analyze_tensor(element)

        if self.is_builtin_class(element):
            return self.analyze_builtin(element)

        msg = f"Type {type(element)} is unsupported at analyze_element"
        print_error_log(msg)
        raise NotImplementedError(msg)

    def analyze_tensor(self, arg):
        api_args = self.api_name + '.' + str(self.args_num)
        check_path_before_create(self.full_save_path)
        create_directory(self.full_save_path)
        file_path = os.path.join(self.full_save_path, f'{api_args}.pt')
        pt_path = write_pt(file_path, arg.contiguous().cpu().detach())
        self.args_num += 1
        return pt_path

    def analyze_builtin(self, arg):
        self.args_num += 1
