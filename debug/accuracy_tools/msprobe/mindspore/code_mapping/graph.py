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


from typing import List, Dict, Union


class GraphNode:
    def __init__(self, name: str, pos: int = -1, unique_name: str = "", operator_name: str = "",
                 return_variable: str = "", return_value: str = "",
                 var_inputs: List[str] = None, has_constant_input: bool = False,
                 unique_id: str = "", scope: str = "", code_info: List[str] = None,
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

