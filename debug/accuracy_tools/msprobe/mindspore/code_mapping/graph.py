from typing import List, Dict, Union
from collections import defaultdict, deque

class GraphNode:
    def __init__(self, name: str, pos: int = -1, unique_name: str = "", operator_name: str = "", return_variable: str = "", return_value: str = "",
                 var_inputs: List[str] = None, has_constant_input: bool = False, unique_id: str="", scope: str = "", code_info: List[str] = None,
                 is_subgraph: bool = False, attrs: Union[Dict[str, str], List[str]] = None):
        self.name = name
        self.unique_name = unique_name
        self.pos = pos
        self.operator_name = operator_name
        self.return_variable = return_variable
        self.return_value = return_value
        self.var_inputs = var_inputs if var_inputs else []
        self.has_constant_input = has_constant_input
        self.unique_id = unique_id
        self.scope = scope
        self.code_info = code_info if code_info else []
        self.attrs = attrs if attrs else ({} if not is_subgraph else [])
        self.nodes = {}  # Internal nodes if this is a subgraph
        self.predecessors = []  # Predecessor nodes
        self.successors = []    # Successor nodes
        self.is_subgraph = is_subgraph

    def trace_back_ancestors(self, ancestors: List[str], visited: Dict[str, bool], parser) -> None:
        if visited[self.unique_name]:
            return
        visited[self.unique_name] = True
        ancestors.append(self.unique_name)
        for predecessor in self.predecessors:
            predecessor.trace_back_ancestors(ancestors, visited, parser)


class Graph:
    def __init__(self, nodes):
        self.nodes = set(nodes.values())

    def topological_sort(self):
        # 创建邻接表和入度表
        nodes = self.nodes
        in_degree = {node: len(node.predecessors) for node in nodes} 
        
        # 初始化队列，将所有入度为 0 的节点加入队列
        queue = deque([node for node in nodes if in_degree[node] == 0])
        topo_order = []

        # Kahn算法的拓扑排序
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            
            for successor in node.successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return topo_order

    def find_independent_nodes(self, subset_nodes):
        # 获取整个图的拓扑排序

        topo_order = self.topological_sort()
        
        # 将子集节点记录为集合，方便查找
        subset_set = set(subset_nodes)
        
        # 追踪哪些子集节点有被访问过
        visited = set()
        
        # 筛选出不被其他子集节点依赖的节点
        independent_nodes = []
        
        # 按照拓扑排序遍历
        for node in topo_order:
            if node in subset_set:
                # 如果该节点在子集中，检查它是否已经被访问
                if node not in visited:
                    independent_nodes.append(node)
                # 将该节点指向的所有邻居标记为访问过（被依赖过）
                for successor in node.successors:
                    if successor in subset_set:
                        visited.add(successor)
        return independent_nodes


def find_boundary_nodes(nodes, domain_level):
    domain_structure = defaultdict(lambda: {'boundary': {'upper': set(), 'lower': set()}, 'nodes': set()})

    for node in nodes:
        if node.scope.startswith("Gradient"):
            continue
        node_new_scope = node.scope.split('/')
        if domain_level <= len(node_new_scope) - 1:  # 确保不使用最后一级
            current_domain = '/'.join(node_new_scope[:domain_level])
            domain_structure[current_domain]['nodes'].add(node)

    for domain, data in domain_structure.items():
        # 遍历域内的节点，寻找上边界和下边界
        for node in data['nodes']:
            if not node.operator_name.startswith("Prim"):
                continue
            node_scope = node.scope.split('/')
            for succ in node.successors:
                succ_scope = succ.scope.split('/')
                if succ.scope.startswith("Gradient") or len(succ_scope) == 2:
                    continue
                if (succ.operator_name != "Param" and succ.operator_name != "Constant") and node_scope[:domain_level] != succ_scope[:domain_level]:
                    data['boundary']['lower'].add(node.name)
            for pred in node.predecessors:
                pred_scope = pred.scope.split('/')
                if (pred.operator_name != "Param" and pred.operator_name != "Constant") and node_scope[:domain_level] != pred_scope[:domain_level]:
                    data['boundary']['upper'].add(node.name)

        # 递归处理子域
        sub_nodes = [node for node in data['nodes'] if len(node.scope) > domain_level]
        if sub_nodes:
            domain_structure[domain].update(find_boundary_nodes(sub_nodes, domain_level + 1))
    return domain_structure
