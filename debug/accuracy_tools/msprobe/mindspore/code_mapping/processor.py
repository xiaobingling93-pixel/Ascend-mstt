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

from msprobe.mindspore.code_mapping.graph_parser import Parser
from msprobe.mindspore.code_mapping.bind import bind_code_info_for_data, write_to_csv
from msprobe.core.common.file_utils import FileOpen


def process(args):
    ir_file_path = args.ir
    with FileOpen(ir_file_path, 'r') as f:
        input_text = f.read()

    parser = Parser()
    parser.parse(input_text)

    nodes = parser.get_nodes()

    bind_result = bind_code_info_for_data(args.dump_data, nodes)
    if bind_result:
        write_to_csv(bind_result, args.output)

