import re
from msprobe.core.grad_probe.constant import GradConst
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import write_csv, check_path_before_create, change_mode
from msprobe.core.common.const import FileCheckConst
import matplotlib.pyplot as plt


def data_in_list_target(data, lst):
    return not lst or len(lst) == 0 or data in lst


def check_numeral_list_ascend(lst):
    if any(not isinstance(item, (int, float)) for item in lst):
        raise Exception("The input list should only contain numbers")
    if lst != sorted(lst):
        raise Exception("The input list should be ascending")


def check_param(param_name):
    if not re.match(GradConst.PARAM_VALID_PATTERN, param_name):
        raise RuntimeError("The parameter name contains special characters.")


def check_str(string, variable_name):
    if not isinstance(string, str):
        raise ValueError(f'The variable: "{variable_name}" is not a string.')

def check_bounds_element(bound):
    return GradConst.BOUNDS_MINIMUM <= bound and bound <= GradConst.BOUNDS_MAXIMUM

def check_bounds(bounds):
    prev = GradConst.BOUNDS_MINIMUM - 1
    for element in bounds:
        if not isinstance(element, (int, float)):
            raise Exception("bounds element is not int or float")
        if not check_bounds_element(element):
            raise Exception("bounds element is out of int64 range")
        if prev >= element:
            raise Exception("bounds list is not ascending")
        prev = element

class ListCache(list):
    threshold = 1000

    def __init__(self, *args):
        super().__init__(*args)
        self._output_file = None

    def __del__(self):
        self.flush()

    def flush(self):
        if len(self) == 0:
            return
        if not self._output_file:
            logger.warning("dumpfile path is not setted")
        write_csv(self, self._output_file)
        logger.info(f"write {len(self)} items to {self._output_file}.")
        self.clear()

    def append(self, data):
        list.append(self, data)
        if len(self) >= ListCache.threshold:
            self.flush()

    def set_output_file(self, output_file):
        self._output_file = output_file


def plt_savefig(fig_save_path):
    check_path_before_create(fig_save_path)
    try:
        plt.savefig(fig_save_path)
    except Exception as e:
        raise RuntimeError(f"save plt figure {fig_save_path} failed") from e
    change_mode(fig_save_path, FileCheckConst.DATA_FILE_AUTHORITY)
