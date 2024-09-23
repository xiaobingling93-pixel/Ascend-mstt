import re


class Trie:
    def __init__(self, type_name=None, has_data=False):
        self.type_name = type_name
        self.call_count_list = []
        self.children = {}
        self.has_data = has_data
        self.node_type = None

    def __repr__(self):
        return (f"Node(type_name={self.type_name}, "
                f"has_data={self.has_data}, call number={len(self.call_count_list)})")

    def insert(self, word, word_type="func"):
        parts = word.split('.')
        """
        xxx, node_name, type_name, execute_num
        etc: Cell.network_with_loss.language_model.encoder.layers.1.attention.out_proj.RowParallelLinear.1
        prefix_name_list: Cell.network_with_loss.language_model.encoder.layers.1.attention
        node_name: out_proj
        type_name: RowParallelLinear
        call_count: 1
        """
        type_name = parts[-2]
        call_count = parts[-1]
        node = self
        prefix_name_list = parts[:-2]

        for name in prefix_name_list:
            if name not in node.children:
                node.children[name] = Trie()
            node = node.children[name]
            if node.type_name is None:
                node.type_name = name

        node.type_name = type_name
        node.has_data = True
        node.call_count_list.append(call_count)
        node.node_type = word_type


def dfs_traverse_and_collect_names(node, path="", result=None):
    if result is None:
        result = []

    if node.has_data:
        for count in node.call_count_list:
            result.append(f"{path}.{node.type_name}.{count}")

    for child_name, child_node in node.children.items():
        new_path = f"{path}.{child_name}" if path else child_name
        dfs_traverse_and_collect_names(child_node, new_path, result)

    return result


def dfs_traverse_and_collect_converted_names(node, mapping, path="", mapping_path="", result=None):
    if result is None:
        result = {}
    if node is None:
        return result

    type_name = node.type_name
    if node.has_data:
        for count in node.call_count_list:
            origin_name = f"{path}.{count}" if node.node_type == "Cell" else f"{path}.{type_name}.{count}"
            mapping_name = f"{mapping_path}.{count}" if node.node_type == "Cell" else f"{mapping_path}.{type_name}.{count}"
            result[origin_name] = mapping_name

    name_mapping = mapping.get(type_name, {})

    for child_name, child_node in node.children.items():
        new_path = f"{path}.{child_name}" if path else child_name
        converted_name = name_mapping.get(child_name, child_name)
        new_mapping_path = f"{mapping_path}.{converted_name}" if mapping_path else converted_name
        dfs_traverse_and_collect_converted_names(
            child_node, mapping, new_path, new_mapping_path, result)

    return result


def get_mapping_list(ms_tree, mapping):
    ms_pt_mapping = dfs_traverse_and_collect_converted_names(ms_tree, mapping=mapping)
    mapping_list = []
    for ms_name, pt_name in ms_pt_mapping.items():
        pt_name = re.sub(r"^Cell", "Module", pt_name)
        mapping_list.append((ms_name, pt_name))
    return mapping_list


def get_prefix_mapping(scope_list):
    """layer name to layer name.class_name"""
    layer_mapping = {}
    for name, v in scope_list.items():
        origin_data = v.get("origin_data")
        if not origin_data.startswith(("Cell", "Module")):
            continue
        name_list = name.split('.')
        prefix_name_list = name_list[:-2] + [name_list[-1]]
        prefix_name = '.'.join(prefix_name_list)
        layer_mapping[prefix_name] = name
    return layer_mapping


def get_layer_mapping(ms_scope_list, pt_scope_list, mapping):

    # 1. get layer prefix to full name mapping
    # ect: Cell.network_with_loss.language_model.embedding.3 : Cell.network_with_loss.language_model.embedding.Embedding.3
    ms_prefix2fullname = get_prefix_mapping(ms_scope_list)

    # 2. build trie tree
    ms_tree = Trie(type_name="Cell")
    for k, r in ms_scope_list.items():
        origin_data_name = r.get('origin_data')
        data_type = origin_data_name.split('.')[0]
        ms_tree.insert(k, data_type)
    msname2ptname = get_mapping_list(ms_tree, mapping)

    # 3. get pt layer prefix to full name mapping
    # ect: Module.network_with_loss.language_model.embedding.3 : Module.network_with_loss.language_model.embedding.Embedding.3
    pt_prefix2fullname = get_prefix_mapping(pt_scope_list)

    final_mapping = []
    for ms_name, pt_name in msname2ptname:
        final_ms_name = ms_name
        final_pt_name = pt_name
        # cell
        if ms_name in ms_prefix2fullname:
            final_ms_name = ms_prefix2fullname.get(ms_name)
            final_pt_name = pt_prefix2fullname.get(pt_name, None)
        # func
        elif final_ms_name in ms_scope_list:
            final_ms_name = ms_scope_list.get(ms_name)['origin_data']
            # remove forward/backward
            final_ms_name = '.'.join(final_ms_name.split('.')[:-1])
            final_pt_name = pt_scope_list.get(pt_name, None)
            if final_pt_name:
                final_pt_name = final_pt_name['origin_data']
                final_pt_name = '.'.join(final_pt_name.split('.')[:-1])
        else:
            continue
        final_mapping.append((final_ms_name, final_pt_name))

    return final_mapping
