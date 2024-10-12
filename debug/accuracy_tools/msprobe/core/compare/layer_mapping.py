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

import os

from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import load_json, load_yaml, save_yaml
from msprobe.core.common.utils import (add_time_with_yaml, detect_framework_by_dump_json,
                                       get_stack_construct_by_dump_json_path, CompareException)
from msprobe.core.compare.data_scope_parser import get_dump_data_items


class LayerTrie:
    def __init__(self, type_name=None, framework=None):
        self.type_name = type_name
        self.data_items = []
        self.children = {}
        self.framework = framework

    def __repr__(self):
        return f"Layer(type_name={self.type_name}, data_number={len(self.data_items)})"

    def get(self, name):
        return self.children.get(name)

    def insert(self, data_item):
        parts = data_item.full_scope.split(Const.SEP)
        node = self
        scope_name_list = parts[Const.RIGHT_MOVE_INDEX:]

        for name in scope_name_list:
            if name not in node.children:
                node.children[name] = LayerTrie(name)
            node = node.children[name]
        node.data_items.append(data_item)
        node.type_name = data_item.type_name

    def query_scope(self, data_item, mapping=None):
        if not mapping:
            mapping = {}
        new_scope = Const.TOP_LAYER
        scope_list = data_item.full_scope.split(Const.SEP)
        node = self

        for idx in range(len(scope_list) - 1):
            type_name = node.type_name
            name_mapping = mapping.get(type_name, {})
            child_node_name = scope_list[idx + 1]
            converted_name = name_mapping.get(child_node_name, child_node_name)
            new_scope = new_scope + Const.SEP + converted_name
            child_node = node.get(child_node_name)
            node = child_node
        index = -1
        for idx, child in enumerate(node.data_items):
            if data_item.data_name == child.data_name:
                index = idx
        return new_scope, index

    def query_data(self, scope, index, default_value=None):
        parts = scope.split(Const.SEP)
        node = self
        scope_name_list = parts[1:]

        for name in scope_name_list:
            if name not in node.children:
                return default_value
            node = node.children[name]
        if index >= len(node.data_items):
            return default_value
        return node.data_items[index]

    def save_to_yaml(self, output_path):
        result = {
            f"{self.type_name} @ {self}": self.convert_to_dict(self)}
        file_name = add_time_with_yaml(f"{self.framework}_tree.yaml")
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        save_yaml(file_path, result)

    def convert_to_dict(self, node):
        result = {}
        result["data_item"] = [node.data_name for node in node.data_items]
        for child_key, child_node in node.children.items():
            key = f"{child_key} @ {child_node}"
            result[key] = self.convert_to_dict(child_node)
        return result


def get_data_items_and_tree(dump_json_path, output_path):
    framework = detect_framework_by_dump_json(dump_json_path)
    stack, construct = get_stack_construct_by_dump_json_path(dump_json_path)
    dump = load_json(dump_json_path)
    dump_data_items = get_dump_data_items(dump, stack, construct, framework, output_path)
    root = LayerTrie(type_name=Const.TOP_LAYER, framework=framework)
    for data_item in dump_data_items:
        root.insert(data_item)
    if output_path:
        root.save_to_yaml(output_path)
    return dump_data_items, root


def convert_data_item(npu_tree, bench_tree, npu_data_item, mapping):
    new_scope, index = npu_tree.query_scope(npu_data_item, mapping)
    bench_data_item = bench_tree.query_data(new_scope, index)
    return bench_data_item


def update_keys_in_place(d):
    """
    This function is used to compare and maintain compatibility between the old and new versions.
    In the old version, 'Cell' was used as the top layer name, while the new version uses 'TopLayer'.
    """
    cell_value = d.pop(Const.CELL, None)

    if cell_value is not None:
        d[Const.TOP_LAYER] = cell_value


def convert_data_items(npu_tree, bench_tree, npu_data_items, mapping):
    update_keys_in_place(mapping)
    api_mapping = {}
    for npu_data_item in npu_data_items:
        bench_data_item = convert_data_item(npu_tree, bench_tree, npu_data_item, mapping)
        bench_name = bench_data_item.data_name if bench_data_item else ""
        npu_name = npu_data_item.data_name
        api_mapping[npu_name] = bench_name
    return api_mapping


def generate_api_mapping_by_layer_mapping(npu_json_path, bench_json_path, layer_mapping_path, output_path=None):
    npu_data_items, npu_root = get_data_items_and_tree(npu_json_path, output_path)
    _, bench_root = get_data_items_and_tree(bench_json_path, output_path)
    mapping = load_yaml(layer_mapping_path)
    api_mapping = convert_data_items(npu_root, bench_root, npu_data_items, mapping)
    if output_path:
        file_name = add_time_with_yaml("api_mapping.yaml")
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        save_yaml(file_path, api_mapping)
    return api_mapping


def generate_index_set(item, prefix="", depth=0, max_depth=10):
    if depth > max_depth:
        logger.error(f"parse index exceeds the recursion limit.")
        raise CompareException(CompareException.RECURSION_LIMIT_ERROR)
    result = set()
    if isinstance(item, list):
        for idx, value in enumerate(item):
            pre =  f"{prefix}.{idx}" if prefix else str(idx)
            result.update(generate_index_set(value, pre, depth+1, max_depth))
    else:
        result.add(prefix)
    return result


def generate_file_mapping(npu_json_path, bench_json_path, api_list, output_path=None):
    def get_input(data):
        input_list = data.get(Const.INPUT_ARGS)
        if not input_list:
            input_list = data.get(Const.INPUT)
        return input_list

    def generate_input_output_index_set(data, name):
        data_item = data.get(name)
        inputs = get_input(data_item)
        outputs = data_item.get(Const.OUTPUT)
        input_index_set = generate_index_set(inputs)
        output_index_set = generate_index_set(outputs)
        return input_index_set, output_index_set

    def get_common_index_list(npu_index_set, bench_index_set):
        common_index = npu_index_set & bench_index_set
        res = sorted(common_index, key=lambda x: [int(i) for i in x.split(Const.SEP)])
        return res

    def combine_data_name_and_index(npu_name, bench_name, index_list, input_output):
        res = {}
        for index in index_list:
            k = Const.SEP.join([npu_name, input_output, index])
            v = Const.SEP.join([bench_name, input_output, index])
            res[k] = v
        return res

    npu_data = load_json(npu_json_path).get("data", {})
    bench_data = load_json(bench_json_path).get("data", {})
    data_mapping = {}
    for npu_name, bench_name in api_list.items():
        if not bench_name or not npu_name:
            continue
        npu_input_index_set, npu_output_index_set = generate_input_output_index_set(npu_data, npu_name)
        bench_input_index_set, bench_output_index_set = generate_input_output_index_set(bench_data, bench_name)
        common_input_index_list = get_common_index_list(npu_input_index_set, bench_input_index_set)
        common_output_index_list = get_common_index_list(npu_output_index_set, bench_output_index_set)

        data_mapping.update(combine_data_name_and_index(npu_name, bench_name, common_input_index_list, Const.INPUT))
        data_mapping.update(combine_data_name_and_index(npu_name, bench_name, common_output_index_list, Const.OUTPUT))
    if output_path:
        file_name = add_time_with_yaml("data_mapping")
        file_path = os.path.join(os.path.realpath(output_path), file_name)
        save_yaml(file_path, data_mapping)
    return data_mapping


def generate_data_mapping_by_layer_mapping(input_param, layer_mapping_path, output_path=None):
    npu_json_path = input_param.get("npu_json_path")
    bench_json_path = input_param.get("bench_json_path")
    api_mapping = generate_api_mapping_by_layer_mapping(
        npu_json_path, bench_json_path, layer_mapping_path)
    data_mapping = generate_file_mapping(
        npu_json_path, bench_json_path, api_mapping, output_path)
    return data_mapping
