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


import unittest
from unittest.mock import patch
import pandas as pd

from msprof_analyze.cluster_analyse.recipes.communication_time_sum.communication_time_sum import CommunicationTimeSum
from msprof_analyze.prof_common.constant import Constant

NAMESPACE = "msprof_analyze.cluster_analyse.recipes"


class TestCommunicationGroupMap(unittest.TestCase):

    @patch("msprof_analyze.prof_common.database_service.DatabaseService.query_data")
    def test__mapper_func_should_return_mstx_stats_df(self, mock_query_data):
        data_map = {
            Constant.RANK_ID: 0,
            Constant.PROFILER_DB_PATH: "",
            Constant.ANALYSIS_DB_PATH: ""
        }
        df_dict = {
            "CommAnalyzerTime": pd.DataFrame({
                "hccl_op_name": ["hcom_allGather__648_78_1"],
                "group_name": ["3985311255877281648"],
                "start_timestamp": [1747819038106139.0],
                "elapse_time": [22.29027],
                "transit_time": [0.0],
                "wait_time": [0.0],
                "synchronization_time": [0.0],
                "idle_time": [22.29027],
                "step": ["step6"],
                "type": ["collective"]
            }),
            "CommAnalyzerBandwidth": pd.DataFrame({
                "hccl_op_name": ["hcom_allGather__648_78_1"],
                "group_name": ["3985311255877281648"],
                "transport_type": ["HCCS"],
                "transit_size": [0.00006],
                "transit_time": [0.00134],
                "bandwidth": [0.04780],
                "large_packet_ratio": [0],
                "package_size": [0.00006],
                "count": [1],
                "total_duration": [0.00134],
                "step": ["step6"],
                "type": ["collective"]
            })
        }
        mock_query_data.return_value = df_dict
        recipe = CommunicationTimeSum({})
        time_df, bandwidth_df = recipe._mapper_func(data_map, "CommunicationTimeSum")
        self.assertEqual(time_df.shape, (1, 11))
        self.assertEqual(bandwidth_df.shape, (1, 13))

    @patch("msprof_analyze.prof_common.database_service.DatabaseService.query_data")
    @patch(NAMESPACE + ".communication_time_sum.communication_time_sum.CommunicationTimeSum."
                       "run_communication_group_map_recipe")
    @patch(NAMESPACE + ".base_recipe_analysis.BaseRecipeAnalysis.dump_data")
    @patch(NAMESPACE + ".base_recipe_analysis.BaseRecipeAnalysis.mapper_func")
    def test_run_should_save_db_or_notebook(self, mock_mapper_func, mock_dump_data,
                                            mock_run_communication_group_map_recipe, mock_query_data):
        mock_query_data.return_value = {
            "CommunicationGroupMapping": pd.DataFrame({
                "rank_set": ["(1)"],
                "group_name": ["3985311255877281648"],
            })
        }
        mock_run_communication_group_map_recipe.return_value = True
        mock_mapper_func.return_value = [
            (pd.DataFrame({
                "hccl_op_name": ["hcom_allGather__648_78_1"],
                "group_name": ["3985311255877281648"],
                "start_timestamp": [1747819038106139.0],
                "elapse_time": [22.29027],
                "transit_time": [0.0],
                "wait_time": [0.0],
                "synchronization_time": [0.0],
                "idle_time": [22.29027],
                "step": ["step6"],
                "type": ["collective"],
                "rank_id": [1]
            }),
            pd.DataFrame({
                "hccl_op_name": ["hcom_allGather__648_78_1"],
                "group_name": ["3985311255877281648"],
                "transport_type": ["HCCS"],
                "transit_size": [0.00006],
                "transit_time": [0.00134],
                "bandwidth": [0.04780],
                "large_packet_ratio": [0],
                "package_size": [0.00006],
                "count": [1],
                "total_duration": [0.00134],
                "step": ["step6"],
                "type": ["collective"],
                "rank_id": [1]
            }))
        ]
        expected_communication_time = pd.DataFrame({
            "step": ["step6", "step6"],
            "rank_id": [1, 1],
            "hccl_op_name": ["hcom_allGather__648_78_1", "Total Op Info"],
            "group_name": ["3985311255877281648", "3985311255877281648"],
            "start_timestamp": [1747819038106139.0, 0.0],
            "elapsed_time": [22.29027, 22.29027],
            "transit_time": [0.0, 0.0],
            "wait_time": [0.0, 0.0],
            "synchronization_time": [0.0, 0.0],
            "idle_time": [22.29027, 22.29027],
            "synchronization_time_ratio": [0.0, 0.0],
            "wait_time_ratio": [0.0, 0.0],

        })
        expected_communication_bandwidth = pd.DataFrame({
            "step": ["step6", "step6"],
            "rank_id": [1, 1],
            "hccl_op_name": ["hcom_allGather__648_78_1", "Total Op Info"],
            "group_name": ["3985311255877281648", "3985311255877281648"],
            "band_type": ["HCCS", "HCCS"],
            "transit_size": [0.00006, 0.00006],
            "transit_time": [0.00134, 0.00134],
            "bandwidth": [0.04780, 0.04480],
            "large_packet_ratio": [0, 0],
            "package_size": [0.00006, 0.00006],
            "count": [1, 1],
            "total_duration": [0.00134, 0.00134],
        })
        params = {Constant.EXPORT_TYPE: Constant.DB}
        recipe = CommunicationTimeSum(params)
        recipe.run(context=None)
        is_equal = (recipe.communication_time.reset_index(drop=True).
                    equals(expected_communication_time.reset_index(drop=True)))
        self.assertTrue(is_equal)
        is_equal = (recipe.communication_bandwidth.reset_index(drop=True).
                    equals(expected_communication_bandwidth.reset_index(drop=True)))
        self.assertTrue(is_equal)