# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import zlib
from dataclasses import asdict
from typing import List

import numpy as np
import torch
from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import path_len_exceeds_limit
from msprobe.core.common.log import logger
from msprobe.core.data_dump.data_processor.base import BaseDataProcessor, ModuleBackwardInputsOutputs, \
    ModuleForwardInputsOutputs, TensorStatInfo
from msprobe.pytorch.common.utils import save_pt, load_pt
from msprobe.pytorch.free_benchmark import FreeBenchmarkCheck, UnequalRow

is_gpu = False
try:
    import torch_npu
except ImportError:
    is_gpu = True


class PytorchDataProcessor(BaseDataProcessor):
    pytorch_special_type = (torch.device, torch.dtype, torch.Size, torch.Tensor)

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.torch_object_key = {
            "device": self.analyze_device_in_kwargs,
            "dtype": self.analyze_dtype_in_kwargs
        }

    @staticmethod
    def get_md5_for_tensor(x):
        if x.dtype == torch.bfloat16:
            x = x.float()
        tensor_bytes = x.cpu().detach().numpy().tobytes()
        crc32_hash = zlib.crc32(tensor_bytes)
        return f"{crc32_hash:08x}"

    @staticmethod
    def analyze_device_in_kwargs(element):
        single_arg = {}
        single_arg.update({'type': "torch.device"})
        if not isinstance(element, str):
            if hasattr(element, "index"):
                device_value = element.type + ":" + str(element.index)
            else:
                device_value = element.type
            single_arg.update({"value": device_value})
        else:
            single_arg.update({"value": element})
        return single_arg

    @staticmethod
    def analyze_dtype_in_kwargs(element):
        return {"type": "torch.dtype", "value": str(element)}

    @staticmethod
    def get_stat_info(data):
        tensor_stat = TensorStatInfo()
        if data.is_meta:
            return tensor_stat
        data_clone = data.detach()
        if data_clone.numel() == 0:
            return tensor_stat
        elif data_clone.dtype == torch.bool:
            tensor_stat.max = True in data_clone
            tensor_stat.min = False not in data_clone
        elif not data_clone.shape:
            tensor_stat.max = tensor_stat.min = tensor_stat.mean = tensor_stat.norm = data_clone.item()
        elif torch.is_complex(data_clone):
            data_np = data_clone.cpu().numpy()
            data_abs = np.abs(data_np)
            tensor_stat.max = np.max(data_abs).item()
            tensor_stat.min = np.min(data_abs).item()
            tensor_stat.mean = np.mean(data_abs).item()
        else:
            if not data_clone.is_floating_point() or data_clone.dtype == torch.float64:
                data_clone = data_clone.float()
            tensor_stat.max = torch._C._VariableFunctionsClass.max(data_clone).item()
            tensor_stat.min = torch._C._VariableFunctionsClass.min(data_clone).item()
            tensor_stat.mean = torch._C._VariableFunctionsClass.mean(data_clone).item()
            tensor_stat.norm = torch._C._VariableFunctionsClass.norm(data_clone).item()
        return tensor_stat

    @staticmethod
    def handle_tensor_extremum_nan_inf(tensor, operator):
        data_clone = tensor.detach()
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

    @staticmethod
    def _analyze_torch_size(arg):
        return {"type": "torch.Size", "value": list(arg)}

    @classmethod
    def get_special_types(cls):
        return super().get_special_types() + cls.pytorch_special_type

    def analyze_single_element(self, element, suffix_stack):
        if suffix_stack and suffix_stack[-1] in self.torch_object_key:
            return self.torch_object_key[suffix_stack[-1]](element)
        if isinstance(element, torch.Size):
            return self._analyze_torch_size(element)
        converted_numpy, numpy_type = self._convert_numpy_to_builtin(element)
        if converted_numpy is not element:
            return self._analyze_numpy(converted_numpy, numpy_type)
        if isinstance(element, torch.Tensor):
            return self._analyze_tensor(element, Const.SEP.join(suffix_stack))
        if isinstance(element, (bool, int, float, str, slice, type(Ellipsis))):
            return self._analyze_builtin(element)
        return {}

    def _analyze_tensor(self, tensor, suffix):
        tensor_stat = self.get_stat_info(tensor)
        tensor_json = {}
        tensor_json.update({'type': 'torch.Tensor'})
        tensor_json.update({'dtype': str(tensor.dtype)})
        tensor_json.update({"shape": tensor.shape})
        tensor_json.update({"Max": tensor_stat.max})
        tensor_json.update({"Min": tensor_stat.min})
        tensor_json.update({"Mean": tensor_stat.mean})
        tensor_json.update({"Norm": tensor_stat.norm})
        tensor_json.update({"requires_grad": tensor.requires_grad})

        if tensor_stat.max is not None:
            if np.isinf(tensor_stat.max) or np.isnan(tensor_stat.max):
                tensor_json['Max_except_inf_nan'] = self.handle_tensor_extremum_nan_inf(tensor, "max")
        if tensor_stat.min is not None:
            if np.isinf(tensor_stat.min) or np.isnan(tensor_stat.min):
                tensor_json['Min_except_inf_nan'] = self.handle_tensor_extremum_nan_inf(tensor, "min")

        if self.config.summary_mode == Const.MD5:
            tensor_md5 = self.get_md5_for_tensor(tensor)
            tensor_json.update({Const.MD5: tensor_md5})
        return tensor_json


class StatisticsDataProcessor(PytorchDataProcessor):
    pass


class TensorDataProcessor(PytorchDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        saved_tensor = tensor.clone().contiguous().detach()
        save_pt(saved_tensor, file_path)
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        return single_arg


class OverflowCheckDataProcessor(PytorchDataProcessor):
    __slots__ = ["cached_tensors_and_file_paths"]

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.has_overflow = False
        self.support_inf_nan = None
        self.cached_inplace_api_info = {}
        self.cached_tensors_and_file_paths = {}
        self.bits_for_overflow = 8
        self.real_overflow_nums = 0
        self.overflow_nums = config.overflow_nums

    @property
    def is_terminated(self):
        if self.overflow_nums == -1:
            return False
        if self.real_overflow_nums >= self.overflow_nums:
            return True
        return False

    def analyze_pre_forward_inplace(self, name, module_input_output: ModuleForwardInputsOutputs):
        self.has_overflow = False
        self._is_support_inf_nan()
        self.cached_inplace_api_info = super().analyze_pre_forward_inplace(name, module_input_output)
        return None

    def analyze_forward_inplace(self, name, module_input_output: ModuleForwardInputsOutputs):
        self._is_support_inf_nan()
        api_info_struct = super().analyze_forward_inplace(name, module_input_output)
        if name in self.cached_inplace_api_info and name in api_info_struct:
            self.cached_inplace_api_info[name].update(api_info_struct[name])
        elif name in api_info_struct:
            self.cached_inplace_api_info = api_info_struct
        self.handle_overflow()
        return self.cached_inplace_api_info if self.has_overflow else None

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self.has_overflow = False
        self._is_support_inf_nan()
        api_info_struct = super().analyze_forward(name, module, module_input_output)
        self.handle_overflow()
        return api_info_struct if self.has_overflow else None

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        self.has_overflow = False
        self._is_support_inf_nan()
        api_info_struct = super().analyze_backward(name, module, module_input_output)
        self.handle_overflow()
        return api_info_struct if self.has_overflow else None

    def handle_overflow(self):
        if not self.support_inf_nan:
            self._analyze_maybe_overflow_flag()
        if self.has_overflow:
            for file_path, tensor in self.cached_tensors_and_file_paths.items():
                save_pt(tensor, file_path)
            self.real_overflow_nums += 1
            if self.overflow_nums != -1 and self.real_overflow_nums >= self.overflow_nums:
                logger.info(f"[{Const.TOOL_NAME}] Reached the preset overflow times, "
                            f"current overflow times: {self.real_overflow_nums}.")
        self.cached_tensors_and_file_paths = {}

    def _is_support_inf_nan(self):
        if self.support_inf_nan is not None:
            return
        try:
            self.support_inf_nan = is_gpu or torch_npu.npu.utils.is_support_inf_nan()
        except Exception:
            logger.warning(f"Unable to determine if the current device supports inf/nan mode, default not supported.")
            self.support_inf_nan = False

    def _analyze_maybe_overflow_flag(self):
        try:
            self.has_overflow = torch_npu.npu.utils.get_npu_overflow_flag()
            if self.has_overflow:
                torch_npu.npu.utils.clear_npu_overflow_flag()
        except Exception as e:
            logger.error(f"Overflow check failed, the current environment may be abnormal.")
            raise RuntimeError(f"overflow check failed") from e

    def _analyze_maybe_overflow_tensor(self, tensor_json):
        if tensor_json['Max'] is None or tensor_json['Min'] is None:
            return
        self.has_overflow = np.isinf(tensor_json['Max']) or np.isnan(tensor_json['Max']) or \
                            np.isinf(tensor_json['Min']) or np.isnan(tensor_json['Min'])

    def _analyze_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        if not path_len_exceeds_limit(file_path):
            self.cached_tensors_and_file_paths.update({file_path: tensor})
        else:
            logger.warning(f'The file path {file_path} length exceeds limit.')
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        if not self.has_overflow and self.support_inf_nan:
            self._analyze_maybe_overflow_tensor(single_arg)
        return single_arg


class FreeBenchmarkDataProcessor(PytorchDataProcessor):

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.checker = FreeBenchmarkCheck(config=config)
        self._return_forward_new_output = None
        self._forward_new_output = None

    def update_iter(self, current_iter):
        super().update_iter(current_iter)
        self.checker.update_iter(current_iter)

    def update_unequal_rows(self, unequal_rows: List[UnequalRow]):
        if not unequal_rows:
            return
        for row in unequal_rows:
            data_dict = asdict(row)
            self.data_writer.write_data_to_csv(
                data_dict.values(),
                data_dict.keys(),
                self.data_writer.free_benchmark_file_path
            )
        return

    def analyze_pre_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self.checker.pre_forward(name, module, self, module_input_output.args, module_input_output.kwargs)

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        new_output, unequal_rows = self.checker.forward(
            name,
            module,
            module_input_output.args,
            module_input_output.kwargs,
            module_input_output.output,
        )
        self.update_unequal_rows(unequal_rows)
        if self.checker.if_fix():
            self._return_forward_new_output = True
            self._forward_new_output = new_output

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        self.checker.backward(name, module, module_input_output.grad_input)


class KernelDumpDataProcessor(PytorchDataProcessor):
    forward_init_status = False
    multi_output_apis = ["_sort_", "npu_flash_attention"]

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)

    def analyze_forward(self, name, module, module_input_output):
        if self.config.is_forward_acl_dump:
            self.forward_acl_dump(name, module, module_input_output)
        else:
            self.dump_mode_backward_acl_dump(name, module, module_input_output)

    def forward_acl_dump(self, name, module, module_input_output):
        if not KernelDumpDataProcessor.forward_init_status:
            KernelDumpDataProcessor.forward_init_status = True
            torch_npu.npu.synchronize()
            torch_npu.npu.init_dump()
            torch_npu.npu.set_dump(self.config.acl_config)
            torch_npu.npu.synchronize()
            if self.op_need_trigger(name):
                module.forward(*module_input_output.args, **module_input_output.kwargs).cpu()
            else:
                module.forward(*module_input_output.args, **module_input_output.kwargs)
            torch_npu.npu.synchronize()
            torch_npu.npu.finalize_dump()
            torch_npu.npu.synchronize()
        KernelDumpDataProcessor.forward_init_status = False
        logger.info("Dump %s op file." % name)

    def acl_backward_dump_status(self, output, grad, module_name):
        if isinstance(output, torch.Tensor):
            output.backward(grad, retain_graph=True)
            return True

        for api_name in KernelDumpDataProcessor.multi_output_apis:
            if api_name in module_name:
                output[0].backward(grad, retain_graph=True)
                return True
        return False

    def dump_mode_backward_acl_dump(self, name, module, module_input_output):
        grad_path = self.config.backward_input.get(name)
        if not KernelDumpDataProcessor.forward_init_status:
            KernelDumpDataProcessor.forward_init_status = True
            output = module.forward(*module_input_output.args, **module_input_output.kwargs)
            pt = load_pt(grad_path)
            grad = pt.to("npu").requires_grad_()
            torch_npu.npu.init_dump()
            torch_npu.npu.set_dump(self.config.acl_config)
            torch_npu.npu.synchronize()
            if not self.acl_backward_dump_status(output, grad, name):
                logger.warning("The output of {} is not of tensor type and cannot be automatically derived. "
                               "you can manually construct a single API backward case for ACL dump.".format(
                    name))
            torch_npu.npu.synchronize()
            torch_npu.npu.finalize_dump()
        KernelDumpDataProcessor.forward_init_status = False
        logger.info("Dump %s op file." % name)

    def op_need_trigger(self, module_name):
        return 'Tensor.__getitem__.' in module_name
