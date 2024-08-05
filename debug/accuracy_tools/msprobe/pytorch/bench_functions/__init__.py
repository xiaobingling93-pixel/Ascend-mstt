import os
from pkgutil import iter_modules
from importlib import import_module

"""
gpu and cpu not implement benchmark function, supplementary benchmarking function implementation
"""

package_path = os.path.dirname(os.path.realpath(__file__))
for _, module_name, _ in iter_modules([package_path]):
    module = import_module(f"{__name__}.{module_name}")
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if callable(attr) and "npu_custom" not in attr_name:
            globals()[attr_name] = attr
