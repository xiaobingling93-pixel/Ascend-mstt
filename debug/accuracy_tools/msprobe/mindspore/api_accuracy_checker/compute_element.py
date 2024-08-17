import os

import mindspore
import torch
import numpy as np

from msprobe.core.common.log import logger
from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.core.common.utils import load_npy
from msprobe.mindspore.api_accuracy_checker.type_mapping import (dtype_str_to_np_type, api_info_type_str_to_type,
                                                                 ms_dtype_to_dtype_str, torch_dtype_to_dtype_str,
                                                                 dtype_str_to_ms_dtype, dtype_str_to_np_dtype,
                                                                 dtype_str_to_torch_dtype, DEFAULT_CONSTRUCT_NP_DTYPE)
from msprobe.mindspore.api_accuracy_checker.utils import check_and_get_from_json_dict, global_context


class MstensorMetaData:
    def __init__(self, dtype, npy_path, maximum, minimum, shape) -> None:
        self.dtype = dtype
        self.npy_path = npy_path
        self.maximum = maximum
        self.minimum = minimum
        self.shape = shape

class ComputeElement:
    def __init__(self, compute_element_info=None, parameter=None):
        if parameter is not None:
            self._init_with_parameter(parameter)
        elif isinstance(compute_element_info, (list, dict)):
            self._init_from_compute_element_info(compute_element_info)
        else:
            logger.error_log_with_exp(
                "ComputeElement.__init__ failed: not init with parameter or compute_element info is not (list, dict)",
                ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))

    def _init_from_compute_element_info(self, compute_element_info):
        '''
        Args:
            compute_element_info: Union[list, dict]
            is_constructed: boolean

        Return:
            void

        init member attributes: self.shape, self.dtype_str, self.parameter
        '''
        if isinstance(compute_element_info, list):
            self.shape = tuple()
            self.dtype_str = "tuple"
            self.parameter = tuple(ComputeElement(compute_element_info=sub_info).get_parameter()
                                   for sub_info in compute_element_info)
        else:
            type_str = check_and_get_from_json_dict(compute_element_info, "type", "type field in api_info.json",
                                                    accepted_type=str, accepted_value=api_info_type_str_to_type.keys())

            if type_str == "mindspore.Tensor":
                self._init_from_mstensor_compute_element_info(compute_element_info)
            else: # type_str in ("slice", "int", "float", "bool")
                value = check_and_get_from_json_dict(compute_element_info, "value", "value field in api_info.json")
                self.shape = tuple()
                self.dtype_str = type_str
                self.parameter = slice(*tuple(value)) if type_str == "slice" else value

    def _init_from_mstensor_compute_element_info(self, compute_element_info):
        '''
            do not load real tensor, only record meta data
        '''
        dtype_str = check_and_get_from_json_dict(compute_element_info, "dtype", "dtype field in api_info.json",
                                            accepted_type=str, accepted_value=dtype_str_to_ms_dtype.keys())
        shape = check_and_get_from_json_dict(compute_element_info, "shape", "shape field in api_info.json",
                                                accepted_type=(list,))
        if global_context.get_is_constructed():
            maximum = check_and_get_from_json_dict(compute_element_info, "Max", "Max field in api_info.json",
                                                    accepted_type=(int, float))
            minimum = check_and_get_from_json_dict(compute_element_info, "Min", "Min field in api_info.json",
                                                accepted_type=(int, float))

            npy_path = None
        else:
            maximum, minimum = None, None
            data_name = check_and_get_from_json_dict(compute_element_info, "data_name",
                                                "data_name field in api_info.json", accepted_type=(str,))
            npy_path = os.path.join(global_context.get_dump_data_dir(), data_name)
        mstensor_meta_data = MstensorMetaData(dtype_str, npy_path, maximum, minimum, shape)
        self.parameter = mstensor_meta_data
        self.dtype_str = dtype_str
        self.shape = tuple(shape)

    def _init_with_parameter(self, parameter):
        self.parameter = parameter
        if isinstance(parameter, mindspore.Tensor):
            self.shape = tuple(parameter.shape)
            self.dtype_str = ms_dtype_to_dtype_str.get(parameter.dtype)
        elif isinstance(parameter, torch.Tensor):
            self.shape = tuple(parameter.shape)
            self.dtype_str = torch_dtype_to_dtype_str.get(parameter.dtype)
        elif isinstance(parameter, (int, float, str, slice, tuple)):
            self.shape = tuple()
            self.dtype_str = "tuple" if isinstance(parameter, tuple) else api_info_type_str_to_type.get(type(parameter))
        else:
            err_msg = "ComputeElement._init_with_parameter failed: " \
                "parameter type is not in (int, float, str, slice, torch.Tensor, mindspore.Tensor)"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))

    def get_parameter(self, get_origin=True, get_mindspore_tensor=True):
        '''
        Args:
            get_origin: boolean
            get_mindspore_tensor: boolean

        Return:
            parameter: Union[int, float, str, slice,tuple,  torch.Tensor, mindspore.Tensor]
        '''
        if isinstance(self.parameter, (int, float, str, slice, torch.Tensor, tuple, mindspore.Tensor)):
            parameter_tmp = self.parameter
        elif isinstance(self.parameter, MstensorMetaData):
            mstensor_meta_data = self.parameter
            ms_dtype = dtype_str_to_ms_dtype.get(mstensor_meta_data.dtype_str)
            if global_context.get_is_constructed():
                np_dtype = dtype_str_to_np_dtype.get(mstensor_meta_data.dtype_str, DEFAULT_CONSTRUCT_NP_DTYPE)
                ndarray = self._construct_ndarray(mstensor_meta_data.shape, mstensor_meta_data.maximum,
                                                  mstensor_meta_data.minimum, np_dtype)
            else:
                ndarray = load_npy(mstensor_meta_data.npy_path)
            parameter_tmp = mindspore.Tensor(ndarray, dtype=ms_dtype)
        else:
            err_msg = "ComputeElement.get_parameter failed: self.parameter type is not in " \
                "(int, float, str, slice, torch.Tensor, mindspore.Tensor, MstensorMetaData)"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))

        # if necessary, do transfer
        if not get_origin and isinstance(parameter_tmp, mindspore.Tensor) and not get_mindspore_tensor:
            parameter = self._transfer_to_torch_tensor(parameter_tmp)
        elif not get_origin and isinstance(parameter, torch.Tensor) and get_mindspore_tensor:
            parameter = self._transfer_to_mindspore_tensor(parameter_tmp)
        else:
            parameter = parameter_tmp

        return parameter

    def get_shape(self):
        return self.shape

    def get_dtype(self):
        return self.dtype_str


    def _transfer_to_torch_tensor(self, ms_tensor):
        '''
        Args:
            ms_tensor: mindspore.Tensor
        Return:
            torch_tensor: torch.Tensor
        '''
        ms_dtype = ms_tensor.dtype
        dtype_str = ms_dtype_to_dtype_str.get(ms_dtype)
        if dtype_str not in dtype_str_to_torch_dtype:
            err_msg = f"ComputeElement._transfer_to_torch_tensor failed: no matching torch dtype for {dtype_str}"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))
        else:
            torch_dtype = dtype_str_to_torch_dtype.get(dtype_str)
        np_ndarray_float64 = ms_tensor.as_type(mindspore.float64).numpy()
        torch_tensor = torch.from_numpy(np_ndarray_float64).to(torch_dtype)
        return torch_tensor

    def _transfer_to_mindspore_tensor(self, torch_tensor):
        '''
        Args:
            torch_tensor: torch.Tensor

        Return:
            ms_tensor: mindspore.Tensor
        '''
        torch_dtype = torch_tensor.dtype
        dtype_str = torch_dtype_to_dtype_str.get(torch_dtype)
        if dtype_str not in dtype_str_to_ms_dtype:
            err_msg = \
                f"ComputeElement._transfer_to_mindspore_tensor failed: no matching mindspore dtype for {dtype_str}"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.UnsupportType))
        else:
            ms_dtype = dtype_str_to_ms_dtype.get(dtype_str)
        np_ndarray_float64 = torch_tensor.to(torch.float64, copy=True).numpy()
        ms_tensor = mindspore.Tensor.from_numpy(np_ndarray_float64).astype(ms_dtype)
        return ms_tensor

    def _convert_inf_to_real_num(self, value, dtype_str):
        if value == float("inf"):
            np_dtype = dtype_str_to_np_type.get(dtype_str, DEFAULT_CONSTRUCT_NP_DTYPE)
            value = np.finfo(np_dtype).max
        elif value == float("-inf"):
            np_dtype = dtype_str_to_np_type.get(dtype_str, DEFAULT_CONSTRUCT_NP_DTYPE)
            value = np.finfo(np_dtype).min
        return value

    def _construct_ndarray(self, shape, maximum, minimum, np_dtype):
        shape = tuple(shape)
        if np_dtype == np.bool_:
            ndarray = np.random.rand(*shape) > 0.5
        else:
            maximum = self._convert_inf_to_real_num(maximum, np_dtype)
            minimum = self._convert_inf_to_real_num(minimum, np_dtype)
            ndarray = np.random.uniform(minimum, maximum, shape).astype(np_dtype)
        return ndarray