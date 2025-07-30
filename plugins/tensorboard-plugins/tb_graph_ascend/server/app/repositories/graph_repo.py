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
from tensorboard.util import tb_logging
logger = tb_logging.get_logger()


class GraphRepo:

    def __init__(self, db_path):
        self.db_path = db_path
        self._initialize_db_connection()

    def _initialize_db_connection(self):
        try:
            print(self.db_path)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.is_db_connected = self.conn is not None
        except:
            logger.error("Failed to connect to database")
            return None
    
    def query_all_nodes(self):
        try:
            query = f"SELECT * FROM tb_nodes"
            with self.conn as c:
                cursor = c.execute(query)
                rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to query nodes: {e}")
            return []
    
    def query_npu_nodes(self):
        try:
            query = f"SELECT * FROM tb_nodes WHERE data_source='NPU'"
            with self.conn as c:
                cursor = c.execute(query)
                rows = cursor.fetchall()
            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to query NPU nodes: {e}")
            return []

    def query_bench_nodes(self):
        try:
            query = f"SELECT * FROM tb_nodes WHERE data_source='Bench'"
            with self.conn as c:
                cursor = c.execute(query)
                rows = cursor.fetchall()
            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to query Bench nodes: {e}")
            return []
