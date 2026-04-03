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

import os
import sys

from tinker.utils.logger import logger


def modellink_import():
    modellink_version = os.getenv('ML_VERSION', "1.1")
    modellink_path = os.getenv('ML_PATH', None)
    if modellink_path is None or not os.path.exists(modellink_path):
        raise RuntimeError("ML_PATH is not set")
    sys.path.append(modellink_path)
    try:
        if modellink_version == "1.0":
            from ascendspeed import megatron_adaptor
        if modellink_version >= "1.0.0":
            import mindspeed_llm
            sys.modules['modellink'] = sys.modules['mindspeed_llm']
        else:
            import modellink
        import megatron
    except ModuleNotFoundError as e:
        raise RuntimeError("ML_PATH is not available. Please make sure it is set correctly!") from e
    from tinker.megatron_patch.patch import patch
    logger.info(f'modellink path {modellink_path}')
    patch()
