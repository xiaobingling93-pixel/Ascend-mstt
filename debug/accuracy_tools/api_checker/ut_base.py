import numpy as np
import os
import mindspore as ms
import torch
from common.utils import Const, dtype_map
from collections import deque
from collections import defaultdict


class UTBase:
    def __init__(self, name, args, kwargs, output=None, real_data=False, stack=None, comparator=None):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.output = output
        self.real_data = real_data
        self.stack = stack
        self.comparator = comparator
    
    @staticmethod
    def convert_list_to_tuple(data):
        for key, value in data.items():
            if not isinstance(value, list):
                continue
            data[key] = tuple(value)
        return data
    
    def insert_into_dict(self, dic, keys, value):
        key = keys.pop(0)
        if not keys:  
            if key in dic:
                dic[key].append(value)
            else:
                dic[key] = [value]
        else:
            if key not in dic:
                dic[key] = defaultdict(list)
            self.insert_into_dict(dic[key], keys, value)
    
    def forward_mindspore_impl(self, *input):
        pass
    
    def forward_pytorch_impl(self, *input):
        pass
    
    def forward_cmp_real(self):
        data_input = self.args
        input_pt = []
        for data in data_input:
            if data.shape:
                input_pt.append(torch.from_numpy(data))
            else:
                origin_data = data.item()
                if data.dtype == np.int64:
                    origin_data = int(origin_data)
                elif data.dtype == np.float64:
                    origin_data = float(origin_data)
                input_pt.append(origin_data)
        
        output_pt = self.forward_pytorch_impl(*input_pt)
        
        if isinstance(output_pt, torch.Tensor):
            output_ms = self.output[0]
            output_pt = output_pt.numpy()
            self.comparator.compare(output_pt, output_ms, self.name + "." + Const.INPUT)
        else:
            for index_output, (output_p, output_ms) in enumerate(zip(output_pt, self.output)):
                output_p = output_p.numpy()
                output_ms = np.load(os.path.join(self.save_path, output_ms))
                self.comparator.compare(output_p, output_ms, self.name + "." + Const.OUTPUT + "." + str(index_output))
    
    def forward_cmp_random(self):
        pass
    
    def compare(self):
        if self.real_data:
            self.forward_cmp_real()
        else:
            self.forward_cmp_random()