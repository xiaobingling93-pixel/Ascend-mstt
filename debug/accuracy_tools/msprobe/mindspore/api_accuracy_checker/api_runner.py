

import mindspore
import torch
from mindspore import ops

from msprobe.mindspore.api_accuracy_checker.compute_element import ComputeElement
from msprobe.mindspore.api_accuracy_checker.const import IS_MINDSPORE_API, IS_TORCH_API, MINT, MINT_FUNCTIONAL
from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.core.common.log import logger
from msprobe.mindspore.api_accuracy_checker.utils import convert_to_tuple


class ApiRunner:
    def __init__(self) -> None:
        self.api_parent_module_mapping = {
            (MINT, IS_MINDSPORE_API): mindspore.mint,
            (MINT, IS_TORCH_API): torch,
            (MINT_FUNCTIONAL, IS_MINDSPORE_API): mindspore.mint.nn.functional,
            (MINT_FUNCTIONAL, IS_TORCH_API): torch.nn.functional
        }

    def __call__(self, inputs, api_name_str, kwargs, gradient_inputs=None,
                 is_forward=True, is_mindspore_api=IS_MINDSPORE_API):
        '''
        Args:
            inputs: List[ComputeElement]
            api_name_str: str
            kwargs: dict
            gradient_inputs: Union[List[ComputeElement], None]
            is_forward: boolean
            is_mindspore_api: boolean

        Return:
            outputs: list[ComputeElement]

        Description:
            run mindspore.mint/torch api
        '''
        api_type_str, api_sub_name = self.get_info_from_name(api_name_str)
        api_instance = self._get_api_instance(api_type_str, api_sub_name, is_mindspore_api)

        self._run_api(api_instance, inputs, kwargs, gradient_inputs, is_forward, is_mindspore_api)

    @classmethod
    def get_info_from_name(cls, api_name_str):
        '''
        Args:
            api_name_str: str, the key of data dict in api_info.json. e.g. "MintFunctional.relu.0"

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

    def _get_api_instance(self, api_type_str, api_sub_name, is_mindspore_api):
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

        api_parent_module = self.api_parent_module_mapping.get((api_type_str, is_mindspore_api))
        module_str = "mindspore.mint." if is_mindspore_api else "torch."
        submodule_str = "nn.functional." if api_type_str == MINT_FUNCTIONAL else ""
        full_api_name = module_str + submodule_str + api_sub_name
        if not hasattr(api_parent_module, api_sub_name):
            err_msg = f"ApiRunner._get_api_instance failed: {full_api_name} is not found"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ApiWrong))

        api_instance = getattr(api_parent_module, api_sub_name)
        if not callable(api_instance):
            err_msg = f"ApiRunner._get_api_instance failed: {full_api_name} is not callable"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ApiWrong))

        return api_instance

    def _run_api(self, api_instance, inputs, kwargs, gradient_inputs, is_forward, is_mindspore_api):
        inputs = tuple(compute_element.get_parameter(get_origin=False, get_mindspore_tensor=is_mindspore_api)
                       for compute_element in inputs)

        if is_forward:
            forward_result = api_instance(*inputs, **kwargs) # can be single tensor or tuple
            forward_result_tuple = convert_to_tuple(forward_result)
            res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in forward_result_tuple]
        else:
            if gradient_inputs is None:
                err_msg = f"ApiRunner._run_api failed: run backward api but gradient_inputs is missing"
                logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))
            gradient_inputs = \
                tuple(compute_element.get_parameter(get_origin=False, get_mindspore_tensor=is_mindspore_api)
                    for compute_element in gradient_inputs)
            if is_mindspore_api:
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