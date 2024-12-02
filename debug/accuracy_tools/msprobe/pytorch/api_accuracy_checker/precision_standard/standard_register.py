from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import absolute_standard_api, binary_standard_api, \
    ulp_standard_api, thousandth_standard_api, BINARY_COMPARE_UNSUPPORT_LIST


class StandardRegistry:
    def __init__(self):
        self.comparison_functions = {}
        self.standard_categories = {
            'absolute_threshold': absolute_standard_api,
            'binary_consistency': binary_standard_api,
            'ulp_compare': ulp_standard_api,
            'thousandth_threshold': thousandth_standard_api
        }

    def _get_standard_category(self, api_name, dtype=None):
        if dtype and dtype not in BINARY_COMPARE_UNSUPPORT_LIST:
            return 'binary_consistency'
        for name, category in self.standard_categories.items():
            if api_name in category:
                return name
        return "benchmark"
    
    def register(self, standard, func):
        self.comparison_functions[standard] = func

    def get_comparison_function(self, api_name, dtype=None):
        standard = self._get_standard_category(api_name, dtype)
        return self.comparison_functions.get(standard)
