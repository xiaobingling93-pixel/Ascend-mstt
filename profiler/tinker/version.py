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

from datetime import datetime, timedelta, timezone
import json
import os

# MAJOR和MAJOR编号目前暂时跟随ModelLink， 如1.2指支持到1.0.RC2， PATCH版本随时更新
PROFILER_VERSION_MAJOR = 1
PROFILER_VERSION_MINOR = 3
PROFILER_VERSION_PATCH = 3

OPTIMIZER_VERSION_MAJOR = 1
OPTIMIZER_VERSION_MINOR = 3
OPTIMIZER_VERSION_PATCH = 0


def profiler_version():
    return f'TinkerProfiler-{PROFILER_VERSION_MAJOR}.{PROFILER_VERSION_MINOR}.{PROFILER_VERSION_PATCH}'


def optimizer_version():
    return f'TinkerOptimizer-{OPTIMIZER_VERSION_MAJOR}.{OPTIMIZER_VERSION_MINOR}.{OPTIMIZER_VERSION_PATCH}'


def dump_task_info(dir_path, infos=None, cm=False):
    now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    version = profiler_version()
    if cm:
        version_info = {
            'version_optimizer': optimizer_version(),
            'task_time': now
        }
    else:
        version_info = {
            'version_profiler': profiler_version(),
            'task_time': now
        }
    if infos:
        version_info.update(infos)
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, f'VERSION_{version}.json'), 'w') as version_file:
        json.dump(version_info, version_file, indent=4)
