# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
import time
import sqlite3
from ..utils.graph_utils import GraphUtils
from tensorboard.util import tb_logging
from ..utils.global_state import SINGLE, NPU, BENCH
logger = tb_logging.get_logger()


class GraphRepo:

    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_db_connection()           

    def _initialize_db_connection(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.is_db_connected = self.conn is not None
        except:
            logger.error("Failed to connect to database")
            return None

    def get_db_connection(self):
        return self.conn

    # DB: 查询配置表信息
    def query_config_info(self):
        try:
            query = f"SELECT * FROM tb_config"
            start = time.perf_counter()
            with self.conn as c:
                cursor = c.execute(query)
                rows = cursor.fetchall()
                
            record = dict(rows[0])
            # 构建最终的 data 对象
            config_info = {
                "microSteps": record.get('micro_steps', 1),
                "tooltips": GraphUtils.safe_json_loads(record.get('tool_tip')),
                "overflowCheck": bool(record.get('overflow_check', 1)),
                "isSingleGraph": not record.get('graph_type') == 'compare',
                "colors": GraphUtils.safe_json_loads(record.get('node_colors')),
                "matchedConfigFiles": [],
                "task": record.get('task', ''),
                "rankNum": record.get('rank_num', 0),
                "stepNum": record.get('step_num', 0),
            }
            end = time.perf_counter()
            print("query_config_info time:", end - start)
            return config_info
        except Exception as e:
            logger.error(f"Failed to query config info: {e}")
            return []

    # DB：查询根节点信息
    def query_root_nodes(self, graph_type, rank, step):
        try:
            type = graph_type if graph_type != SINGLE else NPU
            start = time.perf_counter()
            query = """
            SELECT 
                node_name,
                up_node,
                sub_nodes,
                node_type,
                matched_node_link,
                precision_index,
                overflow_level,
                matched_distributed 
            FROM 
                tb_nodes 
            WHERE   
                step = ?
                AND rank = ? 
                AND data_source = ? 
                AND up_node = '' 
            """
            with self.conn as c:
                cursor = c.execute(query, (step, rank, type))
                rows = cursor.fetchall()
                
            end = time.perf_counter()
            print("query_root_nodes time:", end - start)
            if len(rows) > 0:
                return self.convert_db_to_object(dict(rows[0]))
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to query root nodes: {e}")
            return []
    
    # DB：查询当前节点的所有父节点信息
    def query_up_nodes(self, node_name, graph_type, rank, step):
        try:
            start = time.perf_counter()
            type = graph_type if graph_type != SINGLE else NPU
            # 现根据节点名称查询节点信息，根据up_node字段得到父节点名称
            # 再根据父节点名称查询父节点信息
            # 递归查询父节点，直到根节点
            query = """
                WITH RECURSIVE parent_chain AS (
                    SELECT child.id, child.node_name, child.up_node, child.data_source, child.rank, child.step, 0 AS level
                    FROM 
                        tb_nodes child
                    WHERE  
                        child.step = ?
                        AND child.rank = ?
                        AND child.data_source = ?
                        AND child.node_name = ?

                    UNION ALL

                    SELECT 
                        parent.id, 
                        parent.node_name, 
                        parent.up_node,
                        parent.data_source, 
                        parent.rank, 
                        parent.step, 
                        pc.level + 1
                    FROM 
                        tb_nodes parent
                    INNER JOIN parent_chain pc 
                        ON parent.data_source = pc.data_source
                        AND parent.node_name  = pc.up_node
                        AND parent.rank = pc.rank
                        AND parent.step = pc.step
                    WHERE 
                        pc.up_node IS NOT NULL 
                        AND pc.up_node != ''
                    )
                SELECT 
                    tb_nodes.id,
                    tb_nodes.data_source,
                    tb_nodes.node_name,
                    tb_nodes.up_node,
                    tb_nodes.sub_nodes,
                    tb_nodes.node_type,
                    tb_nodes.matched_node_link,
                    tb_nodes.precision_index,
                    tb_nodes.overflow_level,
                    tb_nodes.matched_distributed
                FROM 
                    tb_nodes
                WHERE 
                    id IN (SELECT id FROM parent_chain)
                ORDER BY (
                    SELECT 
                        level 
                    FROM 
                        parent_chain pc 
                    WHERE 
                        pc.node_name = tb_nodes.node_name) 
                    ASC
            """ 
            with self.conn as c:
                cursor = c.execute(query, (step, rank, type, node_name))
                rows = cursor.fetchall()
                
            up_nodes = {}
            for row in rows:
                dict_row = self.convert_db_to_object(dict(row))
                up_nodes[row['node_name']] = dict_row
            end = time.perf_counter()
            print("query_up_nodes time:", end - start)
            return up_nodes
        except Exception as e:
            logger.error(f"Failed to query up nodes: {e}")
            return {}

    # DB: 查询待匹配节点的信息，构造graph data
    def query_matched_nodes_info(self, npu_node_name, bench_node_name, rank, step):
        try:
            start = time.perf_counter()
            query = """
                SELECT 
                    id,
                    node_name,
                    node_type,
                    up_node,
                    sub_nodes,
                    data_source,
                    input_data,
                    output_data,
                    matched_node_link
                FROM tb_nodes
                WHERE step = ? AND rank = ? AND data_source = ? AND node_name = ?
                """
            npu_nodes = {}
            bench_nodes = {}
            opposite_npu_node_name = GraphUtils.get_opposite_node_name(npu_node_name)
            opposite_bench_node_name = GraphUtils.get_opposite_node_name(bench_node_name)
            # 定义查询参数列表：(graph_type, node_name, target_dict_key)
            queries = [
                (NPU, npu_node_name, 'npu'),
                (NPU, opposite_npu_node_name, 'npu_opposite'),
                (BENCH, bench_node_name, 'bench'),
                (BENCH, opposite_bench_node_name, 'bench_opposite'),
            ]
            # 存储结果的字典
            nodes_dict = {}
            with self.conn as c:
                for graph_type, node_name, key in queries:
                    if not node_name:  # 可选：跳过空 node_name
                        continue
                    cursor = c.execute(query, (step, rank, graph_type, node_name))
                    rows = cursor.fetchall()
                    if rows:
                        node_obj = self.convert_db_to_object(dict(rows[0]))
                        nodes_dict[key] = {node_obj.get('node_name'): node_obj}
                    else:
                        nodes_dict[key] = {}

            npu_nodes = nodes_dict.get('npu', {}) | nodes_dict.get('npu_opposite', {})
            bench_nodes = nodes_dict.get('bench', {}) | nodes_dict.get('bench_opposite', {})    
            result = self.convert_to_graph_json(npu_nodes, bench_nodes)
            end = time.perf_counter()
            print("query_matched_nodes_info time:", end - start)
            return result
        except Exception as e:
            logger.error(f"Failed to query matched nodes info: {e}")
            return self.convert_to_graph_json({}, {})
            
    # DB: 查询待匹配节点及其子节点的信息，递归查询当前节点信息和其所有的子节点信息，一直叶子节点
    def query_node_and_sub_nodes(self, npu_node_name, bench_node_name, rank, step):
        try:
            start = time.perf_counter()
            query = """
                WITH RECURSIVE descendants AS (
                -- 初始节点选择
                SELECT 
                    id,
                    node_name,
                    node_type,
                    up_node,
                    sub_nodes,
                    data_source,
                    input_data,
                    output_data,
                    matched_node_link,
                    node_order,
                    step,
                    rank
                FROM tb_nodes
                WHERE step = ? AND rank = ? AND data_source = ? AND node_name = ?

                UNION ALL

                -- 递归部分
                SELECT 
                    child.id,
                    child.node_name,
                    child.node_type,
                    child.up_node,
                    child.sub_nodes,
                    child.data_source,
                    child.input_data,
                    child.output_data,
                    child.matched_node_link,
                    child.node_order,
                    child.step,
                    child.rank
                FROM descendants d
                JOIN json_each(d.sub_nodes) AS je          -- 将 sub_nodes JSON 数组展开为多行
                JOIN tb_nodes child 
                    ON child.node_name = je.value         -- 子节点名称匹配
                    AND child.step = d.step
                    AND child.rank = d.rank
                    AND child.data_source = d.data_source
                WHERE 
                    d.sub_nodes IS NOT NULL               -- 父节点的 sub_nodes 不为 NULL
                    AND d.sub_nodes != ''               -- 不是空
                    AND d.sub_nodes != '[]' 
                    AND json_type(d.sub_nodes) = 'array'  -- 确保是合法 JSON 数组
            )
            SELECT * FROM descendants
            """ 

            def fetch_and_convert_rows(cursor):
                """
                Helper function to fetch rows from cursor and convert them.
                :param cursor: SQLite cursor object
                :return: Dictionary of nodes keyed by node_name
                """
                nodes = {}
                for row in cursor.fetchall():
                    dict_row = self.convert_db_to_object(dict(row))
                    nodes[row['node_name']] = dict_row
                return nodes

            npu_nodes = {}
            bench_nodes = {}
            opposite_npu_node_name = GraphUtils.get_opposite_node_name(npu_node_name)
            opposite_bench_node_name = GraphUtils.get_opposite_node_name(bench_node_name)
            # 定义查询参数列表：(graph_type, node_name, target_dict_key)
            queries = [
                (NPU, npu_node_name, 'npu'),
                (NPU, opposite_npu_node_name, 'npu_opposite'),
                (BENCH, bench_node_name, 'bench'),
                (BENCH, opposite_bench_node_name, 'bench_opposite'),
            ]
            # 存储结果的字典
            nodes_dict = {}
            with self.conn as c:
                for graph_type, node_name, key in queries:
                    if not node_name:  # 可选：跳过空 node_name
                        continue
                    cursor = c.execute(query, (step, rank, graph_type, node_name))
                    nodes_dict[key] = fetch_and_convert_rows(cursor)
            npu_nodes = nodes_dict.get('npu', {}) | nodes_dict.get('npu_opposite', {})
            bench_nodes = nodes_dict.get('bench', {}) | nodes_dict.get('bench_opposite', {})    
            result = self.convert_to_graph_json(npu_nodes, bench_nodes)
            end = time.perf_counter()
            print("query_node_and_sub_nodes time:", end - start)
            return result
        except Exception as e:
            logger.error(f"Failed to query node and sub nodes: {e}")
            return {'NPU': {}, 'Bench': {}}

    # DB: 查询所有以当前为父节点的子节点
    def query_sub_nodes(self, node_name, graph_type, rank, step):
        try:
            start = time.perf_counter()
            type = graph_type if graph_type != SINGLE else NPU
            query = """
                SELECT 
                    node_name,
                    up_node,
                    sub_nodes,
                    node_type,
                    matched_node_link,
                    precision_index,
                    overflow_level,
                    matched_distributed 
                FROM 
                    tb_nodes 
                WHERE  
                    step = ?
                    AND rank = ?
                    AND data_source = ? 
                    AND up_node = ?
            """
            with self.conn as c:
                cursor = c.execute(query, (step, rank, type, node_name))
                rows = cursor.fetchall()
            sub_nodes = {}
            for row in rows:
                dict_row = self.convert_db_to_object(dict(row))
                sub_nodes[row['node_name']] = dict_row
            end = time.perf_counter()
            print("query_sub_nodes time:", end - start)
            return sub_nodes
        except Exception as e:
            logger.error(f"Failed to query sub nodes: {e}")
            return {}

    # DB: 查询当前节点信息
    def query_node_info(self, node_name, graph_type, rank, step):
        try:
            start = time.perf_counter()
            type = graph_type if graph_type != SINGLE else NPU
            query = """
                SELECT 
                    * 
                FROM 
                    tb_nodes 
                WHERE 
                    step = ?
                    AND rank = ? 
                    AND data_source = ? 
                    AND node_name = ?
            """
            with self.conn as c:
                cursor = c.execute(query, (step, rank, type, node_name))
                rows = cursor.fetchall()
                
            end = time.perf_counter()
            print("query_node_info time:", end - start)
            if len(rows) > 0:
                return self.convert_db_to_object(dict(rows[0]))
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to query node info: {e}")
            return {}
        
    # DB：批量查询节点信息

    def query_nodes_info(self, node_names, graph_type, rank, step):
        try:
            start = time.perf_counter()
            type = graph_type if graph_type != SINGLE else NPU
            query = """
                SELECT 
                    * 
                FROM 
                    tb_nodes 
                WHERE 
                    node_name IN ({}) 
                    AND data_source = ?
                    AND rank = ?
                    AND step = ?
                """.format(','.join(['?'] * len(node_names)))
                
            params = node_names + [type, rank, step]
            with self.conn as c:
                cursor = c.execute(query, params)
                rows = cursor.fetchall()
                
            end = time.perf_counter()
            print("query_nodes_info time:", end - start)
            result = {}
            for row in rows:
                dict_row = self.convert_db_to_object(dict(row))
                result[row['node_name']] = dict_row
            return result
        except Exception as e:
            logger.error(f"Failed to query nodes info: {e}")
            return {}
    
    # DB: 查询已匹配节点列表，未匹配节点列表，所有的节点列表
    def query_all_node_info_in_one(self, rank, step, micro_step):
        try:
            start = time.perf_counter()

            # 单次查询：获取 node_name 和 matched_node_link
            query = """
                SELECT 
                    node_name,
                    data_source,
                    matched_node_link 
                FROM 
                    tb_nodes 
                WHERE 
                    step = ?
                    AND rank = ? 
                    AND (? = -1 OR micro_step_id = ?)
            """
            
            with self.conn as conn:
                cursor = conn.execute(query, (step, rank, micro_step, micro_step))
                rows = cursor.fetchall()

            # 初始化结果
            npu_node_list = []
            bench_node_list = []
            npu_match_node = {}  # {node_name: last_matched_link}
            bench_match_node = {}
            npu_unmatch_node = []
            bench_unmatch_node = []

            # 一次性遍历结果，分类处理
            for row in rows:
                node_name = row['node_name']
                matched_link_str = row['matched_node_link']
                if row['data_source'] == NPU:
                    npu_node_list.append(node_name)
                    # 解析 matched_node_link
                    matched_link = GraphUtils.safe_json_loads(matched_link_str)
                    # 判断是否为有效匹配（非空列表）
                    if isinstance(matched_link, list) and len(matched_link) > 0:
                        npu_match_node[node_name] = matched_link[-1]  # 取最后一个匹配项
                    else:
                        npu_unmatch_node.append(node_name)
                elif row['data_source'] == BENCH:
                    bench_node_list.append(node_name)
                    # 解析 matched_node_link
                    matched_link = GraphUtils.safe_json_loads(matched_link_str)
                    # 判断是否为有效匹配（非空列表）
                    if isinstance(matched_link, list) and len(matched_link) > 0:
                        bench_match_node[node_name] = matched_link[-1]  # 取最后一个匹配项
                    else:
                        bench_unmatch_node.append(node_name)
                else:
                    logger.error(f"Invalid data source: {row['data_source']}")

            end = time.perf_counter()
            print(f"query_all_node_info_in_one time: {end - start:.4f}s")

            return {
                'npu_node_list': npu_node_list,
                'bench_node_list': bench_node_list,
                'npu_match_node': npu_match_node,
                'bench_match_node': bench_match_node,
                'npu_unmatch_node': npu_unmatch_node,
                'bench_unmatch_node': bench_unmatch_node
            }

        except Exception as e:
            logger.error(f"Failed to query all node info: {e}")
            return {
                'npu_node_list': [],
                'bench_node_list': [],
                'npu_match_node': {},
                'bench_match_node': {},
                'npu_unmatch_node': [],
                'bench_unmatch_node': []
            }

    # # DB：根据step rank modify match_node_link查询已经修改的匹配成功的节点关系
    def query_modify_matched_nodes_list(self, rank, step):
        try:
            start = time.perf_counter()
            query = """
                SELECT 
                    node_name,
                    matched_node_link 
                FROM 
                    tb_nodes 
                WHERE 
                    step = ?
                    AND rank = ? 
                    AND modified = 1
                    AND matched_node_link IS NOT NULL
                    AND matched_node_link != '[]'
                    AND matched_node_link != ''
            """
            with self.conn as c:
                cursor = c.execute(query, (step, rank))
                rows = cursor.fetchall()
            result = {}
            for row in rows:
                matched_node_link = GraphUtils.safe_json_loads(row['matched_node_link'])
                node_name = row['node_name']
                if isinstance(matched_node_link, list) and len(matched_node_link) > 0:
                    result[node_name] = matched_node_link[-1]  # 取最后一个匹配项
            end = time.perf_counter()
            print("query_modify_matched_nodes_list time:", end - start)
            return result
        except Exception as e:
            logger.error(f"Failed to query modify matched nodes list: {e}")
            return {}

    # DB：批量更新节点信息
    def update_nodes_info(self, nodes_info, rank, step):
        # 取消匹配和匹配都要走这个逻辑        
        try:
            start = time.perf_counter()
            data = [
                (
                    json.dumps(node['matched_node_link']),
                    json.dumps(node['input_data']),
                    json.dumps(node['output_data']),
                    node['precision_index'],
                    step,
                    rank,
                    node['graph_type'],
                    node['node_name']  # WHERE 条件
                )
                for node in nodes_info
            ]
            query = """
                UPDATE tb_nodes 
                SET 
                    matched_node_link = ?,
                    input_data = ?,
                    output_data = ?,
                    precision_index = ?,
                    modified= 1
                WHERE 
                    step = ?
                    AND rank = ? 
                    AND data_source = ? 
                    AND node_name = ?
            """
            with self.conn as c:
                c.executemany(query, data)
            end = time.perf_counter()
            print("update_nodes_info time:", end - start)
            return True
        except Exception as e:
            logger.error(f"Failed to update nodes info: {e}")
            return False

    def convert_to_graph_json(self, npu_nodes, bench_nodes):
        graph_data = {
            "NPU":{
                "node": npu_nodes,
            },
            "Bench":{
                "node": bench_nodes,
            }
        }
        return graph_data

    def convert_db_to_object(self, data):
        object = {
            "id": data.get('node_name'),
            "node_name": data.get('node_name'),
            "node_type": data.get('node_type'),
            "output_data": GraphUtils.safe_json_loads(data.get('output_data') or "{}"),
            "input_data": GraphUtils.safe_json_loads(data.get('input_data') or "{}"),
            "upnode":data.get('up_node'),
            "subnodes":GraphUtils.safe_json_loads(data.get('sub_nodes') or "[]"),
            "matched_node_link":GraphUtils.safe_json_loads(data.get('matched_node_link') or "[]"),
            "stack_info":GraphUtils.safe_json_loads(data.get('stack_info') or "[]"),
            "micro_step_id": data.get('micro_step_id') or -1,
            "data":{
                "precision_index": data.get('precision_index'),
            },
            "parallel_merge_info": GraphUtils.safe_json_loads(data.get('parallel_merge_info') or "[]"),
            "matched_distributed": GraphUtils.safe_json_loads(data.get('matched_distributed') or "[]"),
            "modified": data.get('modified'),
        }
        return object
    
