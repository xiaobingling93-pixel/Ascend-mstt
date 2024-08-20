from msprobe.mindspore.api_accuracy_checker.compute_element import ComputeElement
from msprobe.core.common.const import Const
from msprobe.mindspore.api_accuracy_checker.const import MsApiAccuracyCheckerConst
from msprobe.mindspore.api_accuracy_checker.utils import check_and_get_from_json_dict
from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.core.common.log import logger

class ApiInfo:
    def __init__(self, api_name):
        self.api_name = api_name
        self.forward_info = None
        self.backward_info = None
        self.has_forward_info = False
        self.has_backward_info = False

    def load_forward_info(self, forward_info_dict):
        self.forward_info = forward_info_dict
        self.has_forward_info = True

    def load_backward_info(self, backward_info_dict):
        self.backward_info = backward_info_dict
        self.has_backward_info = True

    def check_forward_info(self):
        return self.has_forward_info

    def check_backward_info(self):
        return self.has_backward_info

    def get_compute_element_list(self, forward_or_backward, input_or_output):
        '''
        Args:
            forward_or_backward: str, Union["forward", "backward"]
            input_or_output: str,  Union["input", "output"]

        Return:
            compute_element_list: List[ComputeElement]
        '''
        mapping = {
            (Const.FORWARD, Const.INPUT): [self.forward_info, MsApiAccuracyCheckerConst.API_INFO_FORWARD_INPUT,
                                           f"input_args field of {self.api_name} forward api in api_info.json"],
            (Const.FORWARD, Const.OUTPUT): [self.forward_info, MsApiAccuracyCheckerConst.API_INFO_FORWARD_OUTPUT,
                                            f"output field of {self.api_name} forward api in api_info.json"],
            (Const.BACKWARD, Const.INPUT): [self.backward_info, MsApiAccuracyCheckerConst.API_INFO_BACKWARD_INPUT,
                                            f"input field of {self.api_name} backward api in api_info.json"],
            (Const.BACKWARD, Const.OUTPUT): [self.backward_info, MsApiAccuracyCheckerConst.API_INFO_BACKWARD_OUTPUT,
                                             f"output field of {self.api_name} backward api in api_info.json"]
        }
        dict_instance, key, key_desc = mapping.get((forward_or_backward, input_or_output))
        compute_element_info_list = check_and_get_from_json_dict(dict_instance, key, key_desc, accepted_type=list)
        compute_element_list = [ComputeElement(compute_element_info=compute_element_info)
                                for compute_element_info in compute_element_info_list]
        return compute_element_list

    def get_kwargs(self):
        '''
        Return:
            kwargs_compute_element_dict: dict{str: ComputeElement}
        '''
        kwargs_dict = check_and_get_from_json_dict(self.forward_info, MsApiAccuracyCheckerConst.API_INFO_FORWARD_KWARGS,
                                                   "input_kwargs in api_info.json", accepted_type=dict)
        for key_str, compute_element_info in kwargs_dict.items():
            if not isinstance(key_str, str):
                err_msg = "ApiInfo.get_kwargs failed: compute_element_dict key is not a string"
                logger.error_log_with_exp(err_msg,
                                          ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed))
            if not isinstance(compute_element_info, (list, dict)):
                err_msg = "ApiInfo.get_kwargs failed: compute_element_dict value is not a list or dict"
                logger.error_log_with_exp(err_msg,
                                          ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed))
        kwargs_compute_element_dict = {key_str: ComputeElement(compute_element_info=compute_element_info)
                                       for key_str, compute_element_info in kwargs_dict.items()}
        return kwargs_compute_element_dict

