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
import sqlite3
from ..utils.graph_utils import GraphUtils
from tensorboard.util import tb_logging
from ..utils.global_state import SINGLE, NPU
from idlelib.idle_test.test_query import QueryTest
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

    # 查询配置表信息
    def query_config_info(self):
        try:
            query = f"SELECT * FROM tb_config"
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
            return config_info
        except Exception as e:
            logger.error(f"Failed to query config info: {e}")
            return []

    # DB：查询根节点信息
    def query_root_nodes(self, graph_type, rank, step):
        try:
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
                up_node = ''   
                AND data_source = ? 
                AND rank = ? 
                AND step = ?
            """
            with self.conn as c:
                cursor = c.execute(query, (type, rank, step))
                rows = cursor.fetchall()
            if len(rows) > 0:
                return self.convert_db_to_object(dict(rows[0]))
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to query root nodes: {e}")
            return []
    
    # DB：查询当前节点的父节点信息
    def query_up_nodes(self, node_name, graph_type, rank, step):
        try:
            type = graph_type if graph_type != SINGLE else NPU
            # 现根据节点名称查询节点信息，根据up_node字段得到父节点名称
            # 再根据父节点名称查询父节点信息
            query = """
                SELECT 
                    parent.node_name,
                    parent.up_node,
                    parent.sub_nodes,
                    parent.node_type,
                    parent.matched_node_link,
                    parent.precision_index,
                    parent.overflow_level,
                    parent.matched_distributed 
                FROM 
                    tb_nodes child
                LEFT JOIN 
                    tb_nodes parent ON child.up_node = parent.node_name 
                    AND child.data_source=parent.data_source
                    AND child.rank=parent.rank
                    AND child.step=parent.step
                WHERE 
                    child.node_name = ?
                    AND child.data_source= ?
                    AND child.rank= ?
                    AND child.step= ?
            """ 
            with self.conn as c:
                cursor = c.execute(query, (node_name, type, rank, step))
                rows = cursor.fetchall()
            if len(rows) > 0:
                return self.convert_db_to_object(dict(rows[0]))
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to query up nodes: {e}")
            return {}

    # DB: 查询所有以当前为父节点的子节点
    def query_sub_nodes(self, node_name, graph_type, rank, step):
        try:
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
                    up_node = ?
                    AND data_source = ? 
                    AND rank = ? 
                    AND step = ?
            """
            with self.conn as c:
                cursor = c.execute(query, (node_name, type, rank, step))
                rows = cursor.fetchall()
            sub_nodes = {}
            for row in rows:
                dict_row = self.convert_db_to_object(dict(row))
                sub_nodes[row['node_name']] = dict_row
            return sub_nodes
        except Exception as e:
            logger.error(f"Failed to query sub nodes: {e}")
            return {}
    
    # DB：根据graph_type查询节点名称列表
    def query_node_name_list(self, graph_type, rank, step, micro_step):
        try:
            type = graph_type if graph_type != SINGLE else NPU
            query = """
                SELECT 
                    node_name 
                FROM 
                    tb_nodes 
                WHERE 
                    data_source = ? 
                    AND rank = ? 
                    AND step = ?
                    AND (? = -1 OR micro_step_id = ?)
            """
            with self.conn as c:
                cursor = c.execute(query, (type, rank, step, micro_step, micro_step))
                rows = cursor.fetchall()
            return [row['node_name'] for row in rows]
        except Exception as e:
            logger.error(f"Failed to query node names: {e}")
            return []

    # DB: 查询当前节点信息
    def query_node_info(self, node_name, graph_type, rank, step):
        try:
            type = graph_type if graph_type != SINGLE else NPU
            query = """
                SELECT 
                    * 
                FROM 
                    tb_nodes 
                WHERE 
                    node_name = ?
                    AND data_source = ? 
                    AND rank = ? 
                    AND step = ?
            """
            with self.conn as c:
                cursor = c.execute(query, (node_name, type, rank, step))
                rows = cursor.fetchall()
            if len(rows) > 0:
                return self.convert_db_to_object(dict(rows[0]))
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to query node info: {e}")
            return None
    
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
    
