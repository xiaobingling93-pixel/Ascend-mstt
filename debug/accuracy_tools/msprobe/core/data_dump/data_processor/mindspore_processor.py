# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
# ============================================================================

import zlib

import mindspore as ms
from mindspore import mint, ops
from mindspore._c_expression.typing import Number
import numpy as np

from msprobe.core.common.const import Const
from msprobe.core.data_dump.data_processor.base import (BaseDataProcessor, TensorStatInfo,
                                                        ModuleForwardInputsOutputs, ModuleBackwardInputsOutputs)
from msprobe.core.common.file_utils import path_len_exceeds_limit
from msprobe.mindspore.common.utils import convert_bf16_to_fp32, save_tensor_as_npy
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.dump.hook_cell.api_registry import api_register


class MindsporeDataProcessor(BaseDataProcessor):
    mindspore_special_type = tuple([ms.Tensor, Number])

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.mindspore_object_key = {
            "dtype": self.analyze_dtype_in_kwargs
        }

    @staticmethod
    def get_md5_for_tensor(x):
        x = convert_bf16_to_fp32(x)
        tensor_bytes = x.asnumpy().tobytes()
        crc32_hash = zlib.crc32(tensor_bytes)
        return f"{crc32_hash:08x}"

    @staticmethod
    def analyze_dtype_in_kwargs(element):
        return {"type": "mindspore.dtype", "value": str(element)}

    @classmethod
    def get_special_types(cls):
        return super().get_special_types() + cls.mindspore_special_type

    def get_stat_info(self, data):
        tensor_stat = TensorStatInfo()
        if data.numel() == 0:
            return tensor_stat
        elif data.dtype == ms.bool_:
            data_np = data.asnumpy()
            tensor_stat.max = np.max(data_np).item()
            tensor_stat.min = np.min(data_np).item()
        elif not data.shape:
            tensor_stat.max = tensor_stat.min = tensor_stat.mean = tensor_stat.norm = data.item()
        elif data.dtype == ms.complex64 or data.dtype == ms.complex128:
            data_abs = np.abs(data.asnumpy())
            tensor_stat.max = np.max(data_abs).item()
            tensor_stat.min = np.min(data_abs).item()
            tensor_stat.mean = np.mean(data_abs).item()
            tensor_stat.norm = np.linalg.norm(data_abs).item()
        else:
            if not ops.is_floating_point(data):
                data = data.to(ms.float32)
            api_register.norm_inner_op_set_ori_func()
            get_max_value = api_register.mint_ops_ori_attr.get("max", mint.max)
            get_min_value = api_register.mint_ops_ori_attr.get("min", mint.min)
            get_mean_value = api_register.mint_ops_ori_attr.get("mean", mint.mean)
            if hasattr(mint, "norm"):
                get_norm_value = api_register.mint_ops_ori_attr.get("norm", mint.norm)
            else:
                get_norm_value = api_register.functional_ori_attr.get("norm", ops.norm)
            tensor_stat.max = get_max_value(data).item()
            tensor_stat.min = get_min_value(data).item()
            tensor_stat.mean = get_mean_value(data).item()
            tensor_stat.norm = get_norm_value(data).item()
            api_register.norm_inner_op_set_hook_func()
        return tensor_stat

    def analyze_single_element(self, element, suffix_stack):
        if suffix_stack and suffix_stack[-1] in self.mindspore_object_key:
            return self.mindspore_object_key[suffix_stack[-1]](element)

        converted_numpy, numpy_type = self._convert_numpy_to_builtin(element)
        if converted_numpy is not element:
            return self._analyze_numpy(converted_numpy, numpy_type)
        if isinstance(element, Number):
            return self.analyze_dtype_in_kwargs(element)
        if isinstance(element, ms.Tensor):
            return self._analyze_tensor(element, Const.SEP.join(suffix_stack))
        if isinstance(element, (bool, int, float, str, slice, type(Ellipsis))):
            return self._analyze_builtin(element)
        return {}

    def _analyze_tensor(self, tensor, suffix):
        tensor_stat = self.get_stat_info(tensor)
        tensor_json = {
            'type': 'mindspore.Tensor',
            'dtype': str(tensor.dtype),
            'shape': tensor.shape,
            'Max': self.transfer_type(tensor_stat.max),
            'Min': self.transfer_type(tensor_stat.min),
            'Mean': self.transfer_type(tensor_stat.mean),
            'Norm': self.transfer_type(tensor_stat.norm),
        }
        if self.config.summary_mode == Const.MD5:
            tensor_md5 = self.get_md5_for_tensor(tensor)
            tensor_json.update({Const.MD5: tensor_md5})
        return tensor_json


class StatisticsDataProcessor(MindsporeDataProcessor):
    pass


class TensorDataProcessor(MindsporeDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        save_tensor_as_npy(tensor, file_path)
        return single_arg


class OverflowCheckDataProcessor(MindsporeDataProcessor):
    __slots__ = ["cached_tensors_and_file_paths"]

    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.has_overflow = False
        self.cached_tensors_and_file_paths = {}
        self.real_overflow_nums = 0
        self.overflow_nums = config.overflow_nums

    @property
    def is_terminated(self):
        if self.overflow_nums == -1:
            return False
        if self.real_overflow_nums >= self.overflow_nums:
            return True
        return False

    def analyze_forward(self, name, module, module_input_output: ModuleForwardInputsOutputs):
        self.has_overflow = False
        api_info_struct = super().analyze_forward(name, module, module_input_output)
        self.maybe_save_overflow_data()
        return api_info_struct if self.has_overflow else None

    def analyze_backward(self, name, module, module_input_output: ModuleBackwardInputsOutputs):
        self.has_overflow = False
        api_info_struct = super().analyze_backward(name, module, module_input_output)
        self.maybe_save_overflow_data()
        return api_info_struct if self.has_overflow else None

    def maybe_save_overflow_data(self):
        if self.has_overflow:
            for file_path, tensor in self.cached_tensors_and_file_paths.items():
                save_tensor_as_npy(tensor, file_path)
            self.real_overflow_nums += 1
            if self.overflow_nums != -1 and self.real_overflow_nums >= self.overflow_nums:
                logger.info(f"[{Const.TOOL_NAME}] Reached the preset overflow times, "
                            f"current overflow times: {self.real_overflow_nums}.")
        self.cached_tensors_and_file_paths = {}

    def _analyze_maybe_overflow_tensor(self, tensor_json):
        if tensor_json['Max'] is None:
            return
        if np.isinf(tensor_json['Max']) or np.isnan(tensor_json['Max']):
            self.has_overflow = True
        if np.isinf(tensor_json['Min']) or np.isnan(tensor_json['Min']):
            self.has_overflow = True

    def _analyze_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        if not path_len_exceeds_limit(file_path):
            self.cached_tensors_and_file_paths.update({file_path: tensor})
        else:
            logger.warning(f'The file path {file_path} length exceeds limit.')
        single_arg = super()._analyze_tensor(tensor, suffix)
        self._analyze_maybe_overflow_tensor(single_arg)
        single_arg.update({"data_name": dump_data_name})
        return single_arg
