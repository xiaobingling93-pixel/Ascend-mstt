from msprobe.pytorch.api_accuracy_checker.compare.algorithm import get_rel_err_ratio
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.precision_standard.base_standard import BaseCompare


class ThousandthStdCompare(BaseCompare):

    def __init__(self, input_data):
        self.rel_err_orign = input_data.rel_err_orign
        self.compare_column = input_data.compare_column

    def _compute_metrics(self):
        rel_err_thousandth, _ = get_rel_err_ratio(self.rel_err_orign, CompareConst.THOUSAND_RATIO_THRESHOLD)
        metrics = {
            'rel_err_thousandth': rel_err_thousandth
        }
        return metrics
