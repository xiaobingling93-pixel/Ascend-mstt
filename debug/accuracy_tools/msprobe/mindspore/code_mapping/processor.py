import os
import stat
import json
from msprobe.mindspore.code_mapping.graph_parser import Parser
from msprobe.mindspore.code_mapping.bind import bind_code_info_for_data, write_to_csv


def process(args):
    ir_file_path = args.ir
    with open(ir_file_path, 'r') as f:
        input_text = f.read()

    parser = Parser()
    parser.parse(input_text)

    nodes = parser.get_nodes()

    bind_result = bind_code_info_for_data(args.dump_data, nodes)
    if bind_result:
        write_to_csv(bind_result, args.output)

