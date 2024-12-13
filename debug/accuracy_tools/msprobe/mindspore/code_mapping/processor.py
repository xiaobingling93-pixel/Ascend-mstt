import os
import stat
import json
from msprobe.mindspore.code_mapping.graph import Graph, find_boundary_nodes
from msprobe.mindspore.code_mapping.graph_parser import Parser
from msprobe.mindspore.code_mapping.bind import bind_code_info_for_data, write_to_csv


def serialize_domain_structure(domain_structure):
    serialized_structure = {}
    for domain, data in domain_structure.items():
        serialized_structure[domain] = {
            'boundary': {'upper': list(data['boundary']['upper']), 'lower': list(data['boundary']['lower'])},
            'nodes': [node.name for node in data['nodes']]
        }
        # 递归处理子域，避免解析 boundary 和 nodes 部分
        for key in data:
            if key not in ['boundary', 'nodes', 'upper', 'lower']:
                serialized_structure[domain][key] = serialize_domain_structure({key: data[key]})
    return serialized_structure


def process(args):
    ir_file_path = args.ir
    with open(ir_file_path, 'r') as f:
        input_text = f.read()  # 改为安全类

#
    parser = Parser()
    parser.parse(input_text)

    nodes = parser.get_nodes()

    bind_result = bind_code_info_for_data(args.dump_data, nodes)
    if bind_result:
        write_to_csv(bind_result, args.output)

