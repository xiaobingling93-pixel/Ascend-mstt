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

from functools import wraps
from typing import Callable


def register_grad_ready_wrapper(fn: Callable):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        res = fn(self, *args, **kwargs)
        self.params_with_grad.clear()
        return res

    return wrapper


def start_grad_sync_wrapper(fn: Callable):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.communication_handle = None
        self.communication_issued = False
        res = fn(self, *args, **kwargs)
        return res

    return wrapper
