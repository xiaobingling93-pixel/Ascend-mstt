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
    
    def query_graph_nodes(self, graph_type, rank, step):
        """根据graph_type, rank, step 查询当前图所有节点信息"""
        try:
            query = f"SELECT * FROM tb_nodes WHERE data_source='{graph_type}' AND rank='{rank}' AND step='{step}'"
            with self.conn as c:
                cursor = c.execute(query)
                rows = cursor.fetchall()
            root_nodes = ""
            nodes = {}
            for row in rows:
                data = dict(row)
                if not data.get('id'):
                    continue
                if not data.get('up_node'):
                    root_nodes = data.get('node_name')
                node = {
                    "id": data.get('node_name'),
                    "node_type": data.get('node_type'),
                    "output_data": GraphUtils.safe_json_loads(data.get('output_data') or "{}"),
                    "input_data": GraphUtils.safe_json_loads(data.get('input_data') or "{}"),
                    "upnode":data.get('up_node'),
                    "subnodes":GraphUtils.safe_json_loads(data.get('sub_nodes') or "[]"),
                    "matched_node_link":GraphUtils.safe_json_loads(data.get('matched_node_link') or "[]"),
                    "stack_info":GraphUtils.safe_json_loads(data.get('stack_info') or "[]"),
                    "data":{
                        "precision_index": data.get('precision_index'),
                    },
                    "parallel_merge_info": GraphUtils.safe_json_loads(data.get('parallel_merge_info') or "[]"),
                    "matched_distributed": GraphUtils.safe_json_loads(data.get('matched_distributed') or "[]"),
                    "modified": data.get('modified'),
                }
                nodes[data.get('node_name')] = node
            return {
                "root": root_nodes,
                "node": nodes
            }
            
        except Exception as e:
            logger.error(f"Failed to query nodes: {e}")
            return []

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
    # 查询step 
