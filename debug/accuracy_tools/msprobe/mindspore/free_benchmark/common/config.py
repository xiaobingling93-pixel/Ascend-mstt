# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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

from msprobe.mindspore.common.const import FreeBenchmarkConst


class Config:
    is_enable: bool = False
    handler_type = FreeBenchmarkConst.DEFAULT_HANDLER_TYPE
    pert_type = FreeBenchmarkConst.DEFAULT_PERT_TYPE
    stage = FreeBenchmarkConst.DEFAULT_STAGE
    dump_level = FreeBenchmarkConst.DEFAULT_DUMP_LEVEL
    steps: list = []
    ranks: list = []
    dump_path: str = ""
