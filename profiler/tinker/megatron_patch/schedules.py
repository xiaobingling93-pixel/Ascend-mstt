# Copyright (c) 2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.autograd.variable import Variable


def custom_backward(custom_output, custom_grad_output):
    """
    Directly call C++ autograd engine
    """
    if not custom_output.numel() == 1:
        raise RuntimeError("output should be pseudo-'freed' in schedule, to optimize memory")

    if not isinstance(custom_output, torch.Tensor):
        raise RuntimeError("There has an error that output == '%s' ." % type(custom_output).__name__)

    if not isinstance(custom_grad_output, (torch.Tensor, type(None))):
        raise RuntimeError("There has an error that grad_output == '%s' ." % type(custom_grad_output).__name__)

    # To Handle scalar output, will exit if not valid
    if custom_grad_output is None:
        if not custom_output.numel() == 1:
            raise RuntimeError("There has an error that implicit grad requires scalar output .")
        custom_grad_output = torch.ones_like(custom_output, memory_format=torch.preserve_format)

    # Engine for call C++ interface [ see torch/csrc/autograd/python_engine.cpp]
    Variable._execution_engine.run_backward(
        accumulate_grad=True,
        inputs=tuple(),
        keep_graph=True,
        tensors=(custom_output,),
        grad_tensors=(custom_grad_output,),
        create_graph=False,
        allow_unreachable=True)
