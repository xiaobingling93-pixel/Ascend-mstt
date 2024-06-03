import mindspore as ms
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
import torch
import re
import inspect
from ut_base import UTBase
from common.logger import logger


class Common(Cell):
    def __init__(self):
        super().__init__()


class CommonUT(UTBase):
    def __init__(self, name, args, kwargs, output, real_data=False, stack=None, comparator=None):
        super().__init__(name, args, kwargs, output, real_data, stack, comparator)
        
        pattern = re.compile(r"^(.*?)_(.*?)_(.*)$")
        match = pattern.match(self.name)
        
        if match:
            self.name_ms = match.group(1)
            logger.info(f"Common UT compare mindspore api: {self.name_ms}")
            self.name_py = match.group(2)
            self.name = match.group(3)
        else:
            logger.warning("No match UT found")
    
    def forward_mindspore_impl(self, *args):
        output_ms = getattr(P, self.name_ms)()(*args)
        return output_ms
    
    def forward_pytorch_impl(self, *args):
        args_len = len(inspect.getfullargspec(getattr(P, self.name_ms)()).args) - 1
        tensor_list = args[:args_len]
        output_py = getattr(torch,self.name_py)(*tensor_list)
        return output_py