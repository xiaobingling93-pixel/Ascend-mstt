import zlib
import mindspore as ms
import numpy as np

from msprobe.core.common.const import Const
from msprobe.core.data_dump.data_processor.base import BaseDataProcessor, TensorStatInfo
from msprobe.core.common.log import logger
from msprobe.core.common.file_check import path_len_exceeds_limit, change_mode, FileCheckConst
from msprobe.mindspore.dump.hook_cell.wrap_functional import ops_func, mint_ops_func


class MindsporeDataProcessor(BaseDataProcessor):
    mindspore_special_type = tuple([ms.Tensor])
    
    def __init__(self, config, data_writer):
        super().__init__(config, data_writer)
        self.mindspore_object_key = {
            "dtype": self.analyze_dtype_in_kwargs
        }

    @staticmethod
    def get_md5_for_tensor(x):
        if x.dtype == ms.bfloat16:
            x = x.to(ms.float32)
        tensor_bytes = x.asnumpy().tobytes()
        crc32_hash = zlib.crc32(tensor_bytes)
        return f"{crc32_hash:08x}"

    @staticmethod
    def analyze_dtype_in_kwargs(element):
        return {"type": "mindspore.dtype", "value": str(element)}

    @staticmethod
    def get_stat_info(data):
        tensor_stat = TensorStatInfo()
        if data.numel() == 0:
            return tensor_stat
        elif data.dtype == ms.bool_:
            tensor_stat.max = mint_ops_func["max"](data).item()
            tensor_stat.min = mint_ops_func["min"](data).item()
        elif not data.shape:
            tensor_stat.max = tensor_stat.min = tensor_stat.mean = tensor_stat.norm = data.item()
        elif data.dtype == ms.complex64 or data.dtype == ms.complex128:
            data_abs = np.abs(data.asnumpy())
            tensor_stat.max = np.max(data_abs)
            tensor_stat.min = np.min(data_abs)
            tensor_stat.mean = np.mean(data_abs)
            tensor_stat.norm = np.linalg.norm(data_abs)
        else:
            tensor_stat.max = mint_ops_func["max"](data).item()
            tensor_stat.min = mint_ops_func["min"](data).item()
            tensor_stat.mean = mint_ops_func["mean"](data).item()
            tensor_stat.norm = ops_func["norm"](data).item()
        return tensor_stat
    
    @classmethod
    def get_special_types(cls):
        return super().get_special_types() + cls.mindspore_special_type

    def _analyze_tensor(self, tensor, suffix):
        tensor_stat = self.get_stat_info(tensor)
        tensor_json = {}
        tensor_json.update({'type': 'mindspore.Tensor'})
        tensor_json.update({'dtype': str(tensor.dtype)})
        tensor_json.update({"shape": tensor.shape})
        tensor_json.update({"Max": tensor_stat.max})
        tensor_json.update({"Min": tensor_stat.min})
        tensor_json.update({"Mean": tensor_stat.mean})
        tensor_json.update({"Norm": tensor_stat.norm})
        if self.config.summary_mode == Const.MD5:
            tensor_md5 = self.get_md5_for_tensor(tensor)
            tensor_json.update({Const.MD5: tensor_md5})
        return tensor_json

    def analyze_single_element(self, element, suffix_stack):
        if suffix_stack and suffix_stack[-1] in self.mindspore_object_key:
            return self.mindspore_object_key[suffix_stack[-1]](element)

        converted_numpy, numpy_type = self._convert_numpy_to_builtin(element)
        if converted_numpy is not element:
            return self._analyze_numpy(converted_numpy, numpy_type)
        if isinstance(element, ms.Tensor):
            return self._analyze_tensor(element, Const.SEP.join(suffix_stack))

        if isinstance(element, (bool, int, float, str, slice)):
            return self._analyze_builtin(element)
        return None

    def analyze_element(self, element):
        return self.recursive_apply_transform(element, self.analyze_single_element)


class StatisticsDataProcessor(MindsporeDataProcessor):
    pass


class TensorDataProcessor(MindsporeDataProcessor):
    def _analyze_tensor(self, tensor, suffix):
        dump_data_name, file_path = self.get_save_file_path(suffix)
        single_arg = super()._analyze_tensor(tensor, suffix)
        single_arg.update({"data_name": dump_data_name})
        if not path_len_exceeds_limit(file_path):
            if tensor.dtype == ms.bfloat16:
                tensor = tensor.to(ms.float32)
            saved_tensor = tensor.asnumpy()
            np.save(file_path, saved_tensor)
            change_mode(file_path, FileCheckConst.DATA_FILE_AUTHORITY)
        else:
            logger.warning(f'The file path {file_path} length exceeds limit.')
        return single_arg
