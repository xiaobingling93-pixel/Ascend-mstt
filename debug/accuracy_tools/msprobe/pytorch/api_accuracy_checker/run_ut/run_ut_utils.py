import os
import re

from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.file_utils import FileChecker
from msprobe.pytorch.hook_module.wrap_aten import AtenOPTemplate
from msprobe.pytorch.hook_module.wrap_functional import FunctionalOPTemplate
from msprobe.pytorch.hook_module.wrap_npu_custom import NpuOPTemplate
from msprobe.pytorch.hook_module.wrap_tensor import TensorOPTemplate
from msprobe.pytorch.hook_module.wrap_torch import TorchOPTemplate

hf_32_standard_api = ["conv1d", "conv2d"]


class Backward_Message:
    MULTIPLE_BACKWARD_MESSAGE = "Multiple backward is not supported."
    UNSUPPORT_BACKWARD_MESSAGE = "function with out=... arguments don't support automatic differentiation, skip backward."
    NO_BACKWARD_RESULT_MESSAGE = "function backward result is None, skip backward."


class UtDataInfo:
    def __init__(self, bench_grad, device_grad, device_output, bench_output, grad_in, in_fwd_data_list,
                 backward_message, rank=0):
        self.bench_grad = bench_grad
        self.device_grad = device_grad
        self.device_output = device_output
        self.bench_output = bench_output
        self.grad_in = grad_in
        self.in_fwd_data_list = in_fwd_data_list
        self.backward_message = backward_message
        self.rank = rank


def get_validated_result_csv_path(result_csv_path, mode):
    if mode not in ['result', 'detail']:
        raise ValueError("The csv mode must be result or detail")
    result_csv_path_checker = FileChecker(result_csv_path, FileCheckConst.FILE, ability=FileCheckConst.READ_WRITE_ABLE,
                                          file_type=FileCheckConst.CSV_SUFFIX)
    validated_result_csv_path = result_csv_path_checker.common_check()
    if mode == 'result':
        result_csv_name = os.path.basename(validated_result_csv_path)
        pattern = r"^accuracy_checking_result_\d{14}\.csv$"
        if not re.match(pattern, result_csv_name):
            raise ValueError("When continue run ut, please do not modify the result csv name.")
    return validated_result_csv_path


def get_validated_details_csv_path(validated_result_csv_path):
    result_csv_name = os.path.basename(validated_result_csv_path)
    details_csv_name = result_csv_name.replace('result', 'details')
    details_csv_path = os.path.join(os.path.dirname(validated_result_csv_path), details_csv_name)
    details_csv_path_checker = FileChecker(details_csv_path, FileCheckConst.FILE,
                                           ability=FileCheckConst.READ_WRITE_ABLE, file_type=FileCheckConst.CSV_SUFFIX)
    validated_details_csv_path = details_csv_path_checker.common_check()
    return validated_details_csv_path


def exec_api(api_type, api_name, device, args, kwargs):
    if api_type == "Functional":
        torch_api = FunctionalOPTemplate(api_name, str, False)
    if api_type == "Tensor":
        torch_api = TensorOPTemplate(api_name, str, False)
    if api_type == "Torch":
        torch_api = TorchOPTemplate(api_name, str, False)
    if api_type == "Aten":
        torch_api = AtenOPTemplate(api_name, None, False)
    if api_type == "NPU":
        torch_api = NpuOPTemplate(api_name, None, False, device)
    out = torch_api.forward(*args, **kwargs)
    return out
