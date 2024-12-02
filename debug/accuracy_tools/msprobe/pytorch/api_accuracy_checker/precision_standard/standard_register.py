from msprobe.pytorch.api_accuracy_checker.compare.compare_utils import absolute_standard_api, binary_standard_api, \
    ulp_standard_api, thousandth_standard_api, BINARY_COMPARE_UNSUPPORT_LIST


class StandardRegistry:
    """
    Registry class for managing comparison standards and functions.

    This class provides a centralized registry for different comparison standards and their corresponding functions.
    It allows for dynamic registration of comparison functions based on the standard category.

    Attributes:
        comparison_functions (dict): A dictionary mapping standard categories to their corresponding comparison 
        functions.
        standard_categories (dict): A dictionary mapping standard names to their corresponding API categories.

    Methods:
        _get_standard_category(api_name, dtype): Determines the standard category for a given API name and data type.
        register(standard, func): Registers a comparison function for a given standard category.
        get_comparison_function(api_name, dtype): Retrieves the comparison function for a given API name and data type.

    Note:
        The data type is used to determine the standard category if it is not supported by binary comparison.
        If the API name is not found in any standard category, it defaults to the 'benchmark' category.

    See Also:
        BaseCompare: The base class for comparison classes.
    """
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
