# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
# `http://license.coscl.org.cn/MulanPSL2`
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------


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

