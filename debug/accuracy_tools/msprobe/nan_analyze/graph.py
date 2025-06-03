from dataclasses import dataclass
from msprobe.core.common.const import Const
from msprobe.core.common.log import logger
from msprobe.nan_analyse.utils import FileCache, RankPath, is_ignore_op, check_item_anomaly, NanAnalyseConst


@dataclass
class DataNode:
    op_name: str
    rank: int
    dump_path: str
    construct_path: str
    stack_path: str
    inputs: list
    input_args: list
    input_kwargs: dict
    outputs: dict
    layer: int = 0  # 和communication_node的layer保持一致
    sub_layer: int = 0  # 调用顺序，越小表示越先调用

    def __init__(self, op_name, path: RankPath, op_data):
        self.op_name = op_name
        self.rank = path.rank
        self.dump_path = path.dump_path
        self.construct_path = path.construct_path
        self.stack_path = path.stack_path
        self.inputs = op_data.get(Const.INPUT, [])
        self.input_args = op_data.get(Const.INPUT_ARGS, [])
        self.input_kwargs = op_data.get(Const.INPUT_KWARGS, {})
        self.outputs = op_data.get(Const.OUTPUT, {})

    @staticmethod
    def find_stack(stack_info, op_name):
        for item in stack_info:
            if op_name in item[0]:
                return item[1]

    @staticmethod
    def find_complete_construct(construct_info, op_name):
        construct = [op_name]
        while 1:
            op_name = construct_info.get(op_name)
            if not op_name:
                return construct
            construct.insert(0, op_name)

    def is_anomaly(self) -> bool:
        if is_ignore_op(self.op_name):
            return False
        is_input_anomaly = (check_item_anomaly(self.inputs) and check_item_anomaly(self.input_args) and
                            check_item_anomaly(self.input_kwargs))
        is_output_anomaly = check_item_anomaly(self.outputs)
        return (not is_input_anomaly) and is_output_anomaly

    def gen_node_info(self):
        cache = FileCache()
        construct = cache.load_json(self.construct_path)
        stack = cache.load_json(self.stack_path)
        if Const.FORWARD in self.op_name:
            data_info_list = {Const.INPUT_ARGS: self.input_args, Const.INPUT_KWARGS: self.input_kwargs}
        else:
            data_info_list = {Const.INPUT: self.inputs}
        return {'data_info': data_info_list,
                'construct_info': self.find_complete_construct(construct, self.op_name),
                'stack_info': self.find_stack(stack, self.op_name)}


class CommunicationNode:
    def __init__(self, node_id, rank, data: DataNode, layer=0, **kwargs):
        self.node_id = node_id
        self.rank = rank
        self.data = data
        self.layer = layer
        op_name_split = self.data.op_name.split(Const.SEP)
        if len(op_name_split) < 4:
            logger.error(f'invalid op_name: {self.data.op_name}')
            raise RuntimeError(f'invalid op_name: {self.data.op_name}')
        self.api = op_name_split[1]
        self.call_cnt = op_name_split[2]
        self.pre_node = kwargs.get('pre_node')
        self.link_nodes = kwargs.get('link_nodes', {})
        self.dst_nodes = kwargs.get('dst_nodes', {})
        self.src_nodes = kwargs.get('src_nodes', {})
        self.next_nodes = kwargs.get('next_nodes', {})
        self.compute_ops = kwargs.get('compute_ops', [])
        self.type = self._resolve_type()

    def add_next(self, node):
        self.next_nodes[node.node_id] = node
        node.pre_node = self
        node.layer = self.layer + 1

    def add_link(self, node):
        self.link_nodes[node.node_id] = node
        node.link_nodes[self.node_id] = self
        node.layer = self.layer

    def add_dst(self, node):
        self.dst_nodes[node.node_id] = node
        node.src_nodes[self.node_id] = self
        node.layer = self.layer

    def delete(self):
        for node in self.next_nodes.values():
            node.pre_node = None
        for node in self.dst_nodes.values():
            node.src_nodes.pop(self.node_id)
        for node in self.src_nodes.values():
            node.dst_nodes.pop(self.node_id)
        for node in self.link_nodes.values():
            node.link_nodes.pop(self.node_id)
        if self.pre_node:
            self.pre_node.next_nodes.pop(self.node_id)

    def has_nan_inf(self):
        return self.input_has_nan_inf() or check_item_anomaly(self.data.outputs)
    
    def input_has_nan_inf(self):
        return check_item_anomaly(self.data.input_args) or check_item_anomaly(self.data.input_kwargs)

    def find_connected_nodes(self):
        """
        根据 api/类型/入参/调用次数 确定相连接的node所在rank/api/类型/调用次数
        """
        tar_api = NanAnalyseConst.P2P_API_MAPPING.get(self.api, self.api)
        ranks = []
        for dst in [NanAnalyseConst.DST, NanAnalyseConst.DST_GROUP]:
            if dst in self.data.input_kwargs:
                ranks.append(self.data.input_kwargs.get(dst).get('value'))
                break
        for src in [NanAnalyseConst.SRC, NanAnalyseConst.SRC_GROUP]:
            if src in self.data.input_kwargs:
                ranks.append(self.data.input_kwargs.get(src).get('value'))
                break
        if not ranks:
            for item in self.data.input_args:
                if item.get(Const.TYPE) == 'int':
                    ranks.append(item.get('value'))
        ranks.extend(self.data.input_kwargs.get('group', {}).get('group_ranks'))
        return {'ranks': ranks, 'api': f'Distributed.{tar_api}.{self.call_cnt}.forward'}

    def _resolve_type(self):
        for src in [NanAnalyseConst.SRC, NanAnalyseConst.SRC_GROUP]:
            if src in self.data.input_kwargs:
                if self.data.input_kwargs[src].get('value') == self.rank:
                    return NanAnalyseConst.SRC
                else:
                    return NanAnalyseConst.DST
        for dst in [NanAnalyseConst.DST, NanAnalyseConst.DST_GROUP]:
            if dst in self.data.input_kwargs:
                if self.data.input_kwargs[dst].get('value') == self.rank:
                    return NanAnalyseConst.DST
                else:
                    return NanAnalyseConst.SRC
        if self.api in NanAnalyseConst.DIRECTED_API:
            for item in self.data.input_args:
                if item.get(Const.TYPE) == 'int':
                    node_type = NanAnalyseConst.DIRECTED_API[self.api]
                    return node_type if item.get('value') == self.rank else NanAnalyseConst.OPPOSITE_DIR[node_type]
        return NanAnalyseConst.LINK
