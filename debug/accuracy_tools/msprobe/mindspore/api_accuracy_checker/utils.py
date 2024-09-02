from msprobe.core.common.exceptions import ApiAccuracyCheckerException
from msprobe.core.common.const import Const
from msprobe.mindspore.api_accuracy_checker.type_mapping import float_dtype_str_list
from msprobe.mindspore.common.log import logger

def check_and_get_from_json_dict(dict_instance, key, key_description, accepted_type=None, accepted_value=None):
    '''
    Args:
        dict_instance: dict, dict parsed from input json
        key: str
        key_description: str
        accepted_type: tuple
        accepted_value: Union[tuple, list]

    Return:
        value, the corresponding value of "key" in "dict_instance"

    Exception:
        raise ApiAccuracyCheckerException.ParseJsonFailed error when
        1. dict_instance is not a dict
        2. value is None
        3. value is not accepted type
        4. value is not accepted value
    '''
    parse_failed_exception = ApiAccuracyCheckerException(ApiAccuracyCheckerException.ParseJsonFailed)
    if not isinstance(dict_instance, dict):
        logger.error_log_with_exp("check_and_get_from_json_dict failed: input is not a dict", parse_failed_exception)
    value = dict_instance.get(key)
    if value is None:
        logger.error_log_with_exp(f"check_and_get_from_json_dict failed: {key_description} is missing",
                                  parse_failed_exception)
    elif accepted_type is not None and not isinstance(value, accepted_type):
        logger.error_log_with_exp(
            f"check_and_get_from_json_dict failed: {key_description} is not accepted type: {accepted_type}",
            parse_failed_exception)
    elif accepted_value is not None and value not in accepted_value:
        logger.error_log_with_exp(
            f"check_and_get_from_json_dict failed: {key_description} is not accepted value: {accepted_value}",
            parse_failed_exception)
    return value

def convert_to_tuple(input):
    if isinstance(input, (tuple, list)):
        return tuple(input)
    else:
        input_list = [input]
        return tuple(input_list)

def trim_output_compute_element_list(compute_element_list, forward_or_backward):
    '''
    Args:
        compute_element_list: List[ComputeElement]
        forward_or_backward: str, Union["forward", "backward"]
    '''
    trimmed_list = []
    for compute_element in compute_element_list:
        if compute_element.get_parameter() is None or \
            (forward_or_backward == Const.BACKWARD and compute_element.get_dtype() not in float_dtype_str_list):
            # trim case: 1. parameter is None. 2. backward output has non float parameter
            continue
        trimmed_list.append(compute_element)
    return trimmed_list

class GlobalContext:
    def __init__(self):
        self.is_constructed = True
        self.dump_data_dir = ""

    def init(self, is_constructed, dump_data_dir):
        self.is_constructed = is_constructed
        self.dump_data_dir = dump_data_dir

    def get_dump_data_dir(self):
        return self.dump_data_dir

    def get_is_constructed(self):
        return self.is_constructed


global_context = GlobalContext()