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

import sys
from pathlib import Path

current_dir = Path(__file__).resolve()
tinker_parent_dir = current_dir.parent.parent
if tinker_parent_dir not in sys.path:
    sys.path.append(str(tinker_parent_dir))

from tinker.profiler import profile_space
from tinker.search import optimize, cost_model
from tinker.utils.config import parse_args, check_args


# 主函数
def main():
    args = parse_args()
    check_args(args)
    profile_space.run(args)
    optimize.run(args)
    cost_model.run(args)


if __name__ == '__main__':
    main()