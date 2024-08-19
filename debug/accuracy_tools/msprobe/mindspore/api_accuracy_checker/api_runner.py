

import mindspore
import torch
from mindspore import ops

from msprobe.mindspore.api_accuracy_checker.compute_element import ComputeElement
from msprobe.mindspore.api_accuracy_checker.const import (MINDSPORE_PLATFORM, TORCH_PLATFORM, MINT, MINT_FUNCTIONAL,
                                                          FORWARD_API)
from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.core.common.log import logger
from msprobe.mindspore.api_accuracy_checker.utils import convert_to_tuple


class ApiRunner:
    def __init__(self) -> None:
        self.api_parent_module_mapping = {
            (MINT, MINDSPORE_PLATFORM): mindspore.mint,
            (MINT, TORCH_PLATFORM): torch,
            (MINT_FUNCTIONAL, MINDSPORE_PLATFORM): mindspore.mint.nn.functional,
            (MINT_FUNCTIONAL, TORCH_PLATFORM): torch.nn.functional
        }

    def __call__(self, inputs, api_name_str, kwargs, gradient_inputs=None,
                 forward_or_backward=FORWARD_API, api_platform=MINDSPORE_PLATFORM):
        '''
        Args:
            inputs: List[ComputeElement]
            api_name_str: str
            kwargs: dict{str: ComputeElement}
            gradient_inputs: Union[List[ComputeElement], None]
            is_forward: boolean
            is_mindspore_api: boolean

        Return:
            outputs: list[ComputeElement]

        Description:
            run mindspore.mint/torch api
        '''
        api_type_str, api_sub_name = self.get_info_from_name(api_name_str)
        api_instance = self.get_api_instance(api_type_str, api_sub_name, api_platform)

        self.run_api(api_instance, inputs, kwargs, gradient_inputs, forward_or_backward, api_platform)

    @classmethod
    def get_info_from_name(cls, api_name_str):
        '''
        Args:
            api_name_str: str, the key of data dict in api_info.json. e.g. "MintFunctional.relu.0.backward"

        Return:
            api_type_str: str, Union["MintFunctional", "Mint"]
            api_sub_name: str, e.g. "relu"
        '''
        api_name_list = api_name_str.split('.')
        if len(api_name_list) != 4:
            err_msg = f"ApiRunner.get_info_from_name failed: api_name_str: {api_name_str} is not in defined format"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))
        api_type_str, api_sub_name = api_name_list[0], api_name_list[1]
        if api_type_str not in [MINT, MINT_FUNCTIONAL]:
            err_msg = f"ApiRunner.get_info_from_name failed: not mint or mint.nn.functional api"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))

        return api_type_str, api_sub_name

    @classmethod
    def get_api_instance(cls, api_type_str, api_sub_name, api_platform):
        '''
        Args:
            api_type_str: str, Union["MintFunctional", "Mint"]
            api_sub_name: str, e.g. "relu"
            is_mindspore_api: boolean

        Return:
            api_instance: function object

        Description:
            get mindspore.mint/torch api fucntion
            mindspore.mint.{api_sub_name} <--> torch.{api_sub_name}
            mindspore.mint.nn.functional.{api_sub_name} <--> torch.nn.functional.{api_sub_name}
        '''

        api_parent_module = cls.api_parent_module_mapping.get((api_type_str, api_platform))
        module_str = "mindspore.mint." if api_platform == MINDSPORE_PLATFORM else "torch."
        submodule_str = "nn.functional." if api_type_str == MINT_FUNCTIONAL else ""
        full_api_name = module_str + submodule_str + api_sub_name
        if not hasattr(api_parent_module, api_sub_name):
            err_msg = f"ApiRunner.get_api_instance failed: {full_api_name} is not found"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ApiWrong))

        api_instance = getattr(api_parent_module, api_sub_name)
        if not callable(api_instance):
            err_msg = f"ApiRunner.get_api_instance failed: {full_api_name} is not callable"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ApiWrong))

        return api_instance

    @classmethod
    def run_api(cls, api_instance, inputs, kwargs, gradient_inputs, forward_or_backward, api_platform):
        inputs = tuple(compute_element.get_parameter(get_origin=False, tensor_platform=api_platform)
                       for compute_element in inputs)
        kwargs = {key: value.get_parameter(get_origin=False, tensor_platform=api_platform)
                  for key, value in kwargs.items()}

        if forward_or_backward == FORWARD_API:
            forward_result = api_instance(*inputs, **kwargs) # can be single tensor or tuple
            forward_result_tuple = convert_to_tuple(forward_result)
            res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in forward_result_tuple]
        else:
            if gradient_inputs is None:
                err_msg = f"ApiRunner.run_api failed: run backward api but gradient_inputs is missing"
                logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))
            gradient_inputs = \
                tuple(compute_element.get_parameter(get_origin=False, tensor_platform=api_platform)
                    for compute_element in gradient_inputs)
            if api_platform == MINDSPORE_PLATFORM:
                grad_func = ops.GradOperation(get_all=True, sens_param=True)(api_instance)
                backward_result = grad_func(*inputs,  **kwargs, gradient_inputs) # can be single tensor or tuple
                backward_result_tuple = convert_to_tuple(backward_result)
                res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in backward_result_tuple]
            else:
                #set requires_grad
                for tensor in inputs:
                    if hasattr(tensor, "requires_grad"):
                        setattr(tensor, "requires_grad", True)
                forward_result = api_instance(*inputs, **kwargs)
                forward_result.backward(gradient_inputs)
                backward_result_list = []
                for tensor in inputs:
                    if hasattr(tensor, "grad"):
                        backward_result_list.append(getattr(tensor, "grad"))
                res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in backward_result_list]

        return res_compute_element_list


api_runner = ApiRunner()