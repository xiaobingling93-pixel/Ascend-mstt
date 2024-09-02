

import mindspore
import torch
from mindspore import ops

from msprobe.mindspore.api_accuracy_checker.compute_element import ComputeElement
from msprobe.core.common.const import Const, MsCompareConst
from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.mindspore.common.log import logger
from msprobe.mindspore.api_accuracy_checker.utils import convert_to_tuple
from msprobe.mindspore.api_accuracy_checker.type_mapping import float_dtype_str_list, torch_dtype_to_dtype_str


class ApiInputAggregation:
    def __init__(self, inputs, kwargs, gradient_inputs) -> None:
        '''
        Args:
            inputs: List[ComputeElement]
            kwargs: dict{str: ComputeElement}
            gradient_inputs: Union[List[ComputeElement], None]
        '''
        self.inputs = inputs
        self.kwargs = kwargs
        self.gradient_inputs = gradient_inputs

api_parent_module_mapping = {
    (MsCompareConst.MINT, Const.MS_FRAMEWORK): mindspore.mint,
    (MsCompareConst.MINT, Const.PT_FRAMEWORK): torch,
    (MsCompareConst.MINT_FUNCTIONAL, Const.MS_FRAMEWORK): mindspore.mint.nn.functional,
    (MsCompareConst.MINT_FUNCTIONAL, Const.PT_FRAMEWORK): torch.nn.functional
}


class ApiRunner:
    def __call__(self, api_input_aggregation, api_name_str, forward_or_backward=Const.FORWARD,
                 api_platform=Const.MS_FRAMEWORK):
        '''
        Args:
            api_input_aggregation: ApiInputAggregation
            api_name_str: str, e.g. "MintFunctional.relu.0"
            forward_or_backward: str, Union["forward", "backward"]
            api_platform: str, Union["mindspore", "torch"]

        Return:
            outputs: list[ComputeElement]

        Description:
            run mindspore.mint/torch api
        '''
        api_type_str, api_sub_name = self.get_info_from_name(api_name_str)
        api_instance = self.get_api_instance(api_type_str, api_sub_name, api_platform)

        return self.run_api(api_instance, api_input_aggregation, forward_or_backward, api_platform)

    @staticmethod
    def get_info_from_name(api_name_str):
        '''
        Args:
            api_name_str: str, the trimmed key of data dict in api_info.json. e.g. "MintFunctional.relu.0"

        Return:
            api_type_str: str, Union["MintFunctional", "Mint"]
            api_sub_name: str, e.g. "relu"
        '''
        api_name_list = api_name_str.split(Const.SEP)
        if len(api_name_list) != 3:
            err_msg = f"ApiRunner.get_info_from_name failed: api_name_str: {api_name_str} is not in defined format"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))
        api_type_str, api_sub_name = api_name_list[0], api_name_list[1]
        if api_type_str not in [MsCompareConst.MINT, MsCompareConst.MINT_FUNCTIONAL]:
            err_msg = f"ApiRunner.get_info_from_name failed: not mint or mint.nn.functional api"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))

        return api_type_str, api_sub_name

    @staticmethod
    def get_api_instance(api_type_str, api_sub_name, api_platform):
        '''
        Args:
            api_type_str: str, Union["MintFunctional", "Mint"]
            api_sub_name: str, e.g. "relu"
            api_platform: str: Union["mindpore", "torch"]

        Return:
            api_instance: function object

        Description:
            get mindspore.mint/torch api fucntion
            mindspore.mint.{api_sub_name} <--> torch.{api_sub_name}
            mindspore.mint.nn.functional.{api_sub_name} <--> torch.nn.functional.{api_sub_name}
        '''

        api_parent_module = api_parent_module_mapping.get((api_type_str, api_platform))
        module_str = "mindspore.mint." if api_platform == Const.MS_FRAMEWORK else "torch."
        submodule_str = "nn.functional." if api_type_str == MsCompareConst.MINT_FUNCTIONAL else ""
        full_api_name = module_str + submodule_str + api_sub_name
        if not hasattr(api_parent_module, api_sub_name):
            err_msg = f"ApiRunner.get_api_instance failed: {full_api_name} is not found"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ApiWrong))

        api_instance = getattr(api_parent_module, api_sub_name)
        if not callable(api_instance):
            err_msg = f"ApiRunner.get_api_instance failed: {full_api_name} is not callable"
            logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.ApiWrong))

        return api_instance

    @staticmethod
    def run_api(api_instance, api_input_aggregation, forward_or_backward, api_platform):
        inputs = tuple(compute_element.get_parameter(get_origin=False, tensor_platform=api_platform)
                       for compute_element in api_input_aggregation.inputs)
        kwargs = {key: value.get_parameter(get_origin=False, tensor_platform=api_platform)
                  for key, value in api_input_aggregation.kwargs.items()}
        gradient_inputs = api_input_aggregation.gradient_inputs

        if forward_or_backward == Const.FORWARD:
            forward_result = api_instance(*inputs, **kwargs) # can be single tensor or tuple
            forward_result_tuple = convert_to_tuple(forward_result)
            res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in forward_result_tuple]
        else:
            if gradient_inputs is None:
                err_msg = f"ApiRunner.run_api failed: run backward api but gradient_inputs is missing"
                logger.error_log_with_exp(err_msg, ApiAccuracyCheckerException(ApiAccuracyCheckerException.WrongValue))
            gradient_inputs = tuple(compute_element.get_parameter(get_origin=False, tensor_platform=api_platform)
                                    for compute_element in gradient_inputs)
            if api_platform == Const.MS_FRAMEWORK:
                if len(gradient_inputs) == 1:
                    gradient_inputs = gradient_inputs[0]
                def api_with_kwargs(*forward_inputs):
                    return api_instance(*forward_inputs, **kwargs)
                grad_func = ops.GradOperation(get_all=True, sens_param=True)(api_with_kwargs)
                backward_result = grad_func(*inputs, gradient_inputs) # can be single tensor or tuple
                backward_result_tuple = convert_to_tuple(backward_result)
                res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in backward_result_tuple]
            else:
                #set requires_grad
                requires_grad_index = []
                for index, tensor in enumerate(inputs):
                    if isinstance(tensor, torch.Tensor) and \
                        torch_dtype_to_dtype_str.get(tensor.dtype) in float_dtype_str_list:
                        setattr(tensor, "requires_grad", True)
                        requires_grad_index.append(index)
                forward_results = api_instance(*inputs, **kwargs)
                forward_results = convert_to_tuple(forward_results)
                for forward_res, gradient_in in zip(forward_results, gradient_inputs):
                    forward_res.backward(gradient_in)
                backward_result_list = []
                for index in requires_grad_index:
                    backward_result_list.append(getattr(inputs[index], "grad"))
                res_compute_element_list = [ComputeElement(parameter=api_res) for api_res in backward_result_list]

        return res_compute_element_list


api_runner = ApiRunner()