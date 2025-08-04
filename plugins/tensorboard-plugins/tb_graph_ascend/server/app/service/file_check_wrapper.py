# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tensorboard.backend import http_util
from werkzeug.wrappers.request import Request

from ..utils.graph_utils import GraphUtils
from ..utils.global_state import DataType


def check_file_type(func):
    def wrapper(*args, **kwargs):
        if len(args) <= 0:
            raise RuntimeError('Illegal function call, at least 1 parameter is required but got 0')
        request = args[0]
        if not isinstance(request, Request):
            raise RuntimeError('The request "parameter" is not in a format supported by werkzeug')
        meta_data = GraphUtils.safe_json_loads(request.get_data().decode('utf-8'), {}).get('metaData')

        result = {'success': False, 'error': ''}
        if meta_data is None or not isinstance(meta_data, dict):
            result['error'] = 'The query parameter "metaData" is required and must be a dictionary'
            return http_util.Respond(request, result, "application/json")
        data_type = meta_data.get('type')
        run = meta_data.get('run')
        tag = meta_data.get('tag')
        if data_type is None or run is None or tag is None:
            result['error'] = 'The query parameters "type", "run" and "tag" in "metaData" are required'
            return http_util.Respond(request, result, "application/json")
        if data_type not in [e.value for e in DataType]:
            result['error'] = 'Unsupported file type'
            return http_util.Respond(request, result, "application/json")

        return func(*args, **kwargs)

    return wrapper
