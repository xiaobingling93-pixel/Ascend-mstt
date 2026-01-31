# -------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from msprof_analyze.prof_exports.base_stats_export import BaseStatsExport
from msprof_analyze.prof_common.constant import Constant

QUERY_COMMUNICATION_PTA = """
WITH 
band AS (
    SELECT
        hccl_op_name,
        transport_type,
        JSON_OBJECT(
            'Transit Time(ms)', transit_time, 
            'Transit Size(MB)', transit_size, 
            'Bandwidth(GB/s)', bandwidth,
            'Large Packet Ratio', large_packet_ratio
        ) AS band_dict
    FROM CommAnalyzerBandwidth
    WHERE transport_type IN ('SDMA', 'RDMA')
), 
sdma AS (SELECT hccl_op_name, band_dict FROM band WHERE transport_type = 'SDMA'),
rdma AS (SELECT hccl_op_name, band_dict FROM band WHERE transport_type = 'RDMA')

SELECT
    time.hccl_op_name,
    time.group_name,
    time.start_timestamp,
    time.elapse_time,
    time.step,
    time.type,
    sdma.band_dict AS sdma_dict,
    rdma.band_dict AS rdma_dict
FROM CommAnalyzerTime AS time
LEFT JOIN sdma ON time.hccl_op_name = sdma.hccl_op_name
LEFT JOIN rdma ON time.hccl_op_name = rdma.hccl_op_name
"""

QUERY_COMMUNICATION_MINDSPORE = """
WITH 
band AS (
    SELECT
        hccl_op_name,
        transport_type,
        JSON_OBJECT(
            'Transit Time(ms)', transit_time, 
            'Transit Size(MB)', transit_size, 
            'Bandwidth(GB/s)', bandwidth,
            'Large Packet Ratio', large_packet_ratio
        ) AS band_dict
    FROM CommAnalyzerBandwidth
    WHERE transport_type IN ('SDMA', 'RDMA')
), 
sdma AS (SELECT hccl_op_name, band_dict FROM band WHERE transport_type = 'SDMA'),
rdma AS (SELECT hccl_op_name, band_dict FROM band WHERE transport_type = 'RDMA')

SELECT
    time.hccl_op_name,
    time.group_name,
    time.start_timestamp,
    time.elapse_time,
    sdma.band_dict AS sdma_dict,
    rdma.band_dict AS rdma_dict
FROM CommAnalyzerTime AS time
LEFT JOIN sdma ON time.hccl_op_name = sdma.hccl_op_name
LEFT JOIN rdma ON time.hccl_op_name = rdma.hccl_op_name
"""

QUERY_CLUSTER_COMMUNICATION = """
WITH 
band AS (
    SELECT
        hccl_op_name,
        band_type,
        JSON_OBJECT(
            'Transport Type', band_type,
            'Transit Time(ms)', transit_time, 
            'Transit Size(MB)', transit_size, 
            'Bandwidth(GB/s)', bandwidth,
            'Large Packet Ratio', large_packet_ratio
        ) AS band_dict
    FROM {band_table}
    WHERE band_type IN ('SDMA', 'RDMA')
), 
sdma AS (
    SELECT hccl_op_name, band_dict 
    FROM band 
    WHERE band_type = 'SDMA'
),
rdma AS (
    SELECT hccl_op_name, band_dict 
    FROM band 
    WHERE band_type = 'RDMA'
)

SELECT
    group_map.rank_set,
    time.hccl_op_name,
    time.group_name,
    time.start_timestamp,
    time.elapsed_time,
    time.step,
    time.rank_id, 
    sdma.band_dict AS sdma_dict,
    rdma.band_dict AS rdma_dict
FROM {time_table} AS time
JOIN {group_table} AS group_map 
    ON time.group_name = group_map.group_name
LEFT JOIN sdma 
    ON time.hccl_op_name = sdma.hccl_op_name
LEFT JOIN rdma 
    ON time.hccl_op_name = rdma.hccl_op_name
"""

QUERY_CLUSTER_BANDWIDTH = """
SELECT
    step,
    rank_id,
    band_type,
    transit_time,
    transit_size
FROM {band_table}
WHERE band_type IN ('SDMA', 'RDMA')
"""

QUERY_CLUSTER_STEP_TRACE_TIME = """
SELECT *
FROM ClusterStepTraceTime
"""


class CommunicationInfoExport(BaseStatsExport):

    def __init__(self, db_path, is_pta):
        super().__init__(db_path, "None", {})
        self._query = QUERY_COMMUNICATION_PTA if is_pta else QUERY_COMMUNICATION_MINDSPORE


class ClusterAnalysisExport(BaseStatsExport):
    def __init__(self, db_path, data_simplification):
        super().__init__(db_path, "None", {})
        self.cluster_time_table = "ClusterCommunicationTime" if data_simplification else "ClusterCommAnalyzerTime"
        self.cluster_band_table = "ClusterCommunicationBandwidth" if data_simplification \
                                  else "ClusterCommAnalyzerBandwidth"
        self.cluster_group_table = "CommunicationGroupMapping" if data_simplification else "CommunicationGroup"


class ClusterStepTraceTimeExport(ClusterAnalysisExport):
    def __init__(self, db_path):
        super().__init__(db_path, False)
        self._query = QUERY_CLUSTER_STEP_TRACE_TIME


class ClusterCommunicationInfoExport(ClusterAnalysisExport):
    def __init__(self, db_path, data_simplification):
        super().__init__(db_path, data_simplification)
        self._query = QUERY_CLUSTER_COMMUNICATION.format(time_table=self.cluster_time_table,
                                                         band_table=self.cluster_band_table,
                                                         group_table=self.cluster_group_table)


class ClusterBandwidthInfoExport(ClusterAnalysisExport):
    def __init__(self, db_path, data_simplification):
        super().__init__(db_path, data_simplification)
        self._query = QUERY_CLUSTER_BANDWIDTH.format(band_table=self.cluster_band_table)
