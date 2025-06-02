from dataclasses import dataclass
from debug.accuracy_tools.msprobe.core.common.const import Const, CompareConst
# from msprobe.core.common.const import Const, CompareConst
# from msprobe.anomaly_analyze.utils import FileCache, RankPath, is_communication_op, is_ignore_op, check_item_anomaly, AnomalyAnalyseConst
from utils import FileCache, RankPath, is_communication_op, is_ignore_op, check_item_anomaly, AnomalyAnalyseConst


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


class CommunicationNode:
    def __init__(self, node_id, rank, data: DataNode, layer=0, **kwargs):
        self.node_id = node_id
        self.rank = rank
        self.data = data
        self.layer = layer
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
        return any([check_item_anomaly(self.data.inputs), check_item_anomaly(self.data.input_args),
                    check_item_anomaly(self.data.input_kwargs), check_item_anomaly(self.data.outputs)])

    def anomaly_analyze(self):
        pass

    def find_connected_nodes(self):
        """
        根据 api/类型/入参/调用次数 确定相连接的node所在rank/api/类型/调用次数
        """
        api = self._get_api()
        tar_api = AnomalyAnalyseConst.P2P_API_MAPPING.get(api, api)
        call_cnt = self.data.op_name
        ranks = []
        if self.type == AnomalyAnalyseConst.SRC:
            # todo: find ranks from dst or group
            tar_type = AnomalyAnalyseConst.DST
        elif self.type == AnomalyAnalyseConst.DST:
            # todo: find ranks from src or group
            tar_type = AnomalyAnalyseConst.SRC
        else:
            # todo: find ranks from group
            tar_type = AnomalyAnalyseConst.LINK
        return {'ranks': ranks, 'api': f'torch.distributed.{tar_api}', 'call_cnt': call_cnt, 'type': tar_type}

    def _resolve_type(self):
        if 'src' in self.data.input_kwargs:
            if self.data.input_kwargs['src'] == self.rank:
                return AnomalyAnalyseConst.SRC
            else:
                return AnomalyAnalyseConst.DST
        if 'dst' in self.data.input_kwargs:
            if self.data.input_kwargs['dst'] == self.rank:
                return AnomalyAnalyseConst.DST
            else:
                return AnomalyAnalyseConst.SRC
        return AnomalyAnalyseConst.LINK

    def _get_api(self):
        op_name_split = self.data.op_name.lower().split(Const.SEP)
        return op_name_split[op_name_split.index('distributed') + 1]