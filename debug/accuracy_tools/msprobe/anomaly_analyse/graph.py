from dataclasses import dataclass


@dataclass
class DataNode:
    op_name: str
    inputs: list
    input_args: list
    input_kwargs: dict
    outputs: dict

    def is_anomaly(self) -> bool:
        pass


class CommunicationNode:
    def __init__(self, node_id, rank, data, layer=0, **kwargs):
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
        if self.pre_node is not None:
            self.pre_node.next_nodes.pop(self.node_id)

    def has_nan_inf(self):
        pass

    def anomaly_analyze(self):
        pass