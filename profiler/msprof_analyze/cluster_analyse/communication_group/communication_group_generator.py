# -------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

from msprof_analyze.cluster_analyse.communication_group.communication_db_group import CommunicationDBGroup
from msprof_analyze.cluster_analyse.communication_group.communication_db_group import CommunicationDBGroupOptimized
from msprof_analyze.cluster_analyse.communication_group.communication_json_group import CommunicationJsonGroup
from msprof_analyze.prof_common.constant import Constant


SIMPLIFIED = "SIMPLIFIED"
ORIGINAL = "ORIGINAL"


class CommunicationGroupGenerator:

    GROUP_MAP = {
        ORIGINAL: {
            Constant.DB: CommunicationDBGroup,
            Constant.TEXT: CommunicationJsonGroup
        },
        SIMPLIFIED: {
            Constant.DB: CommunicationDBGroupOptimized,
            Constant.TEXT: CommunicationJsonGroup
        }
    }

    def __init__(self, params: dict):
        version = SIMPLIFIED if params.get(Constant.DATA_SIMPLIFICATION) else ORIGINAL
        self.processor = self.GROUP_MAP.get(version).get(params.get(Constant.DATA_TYPE))(params)

    def generate(self):
        return self.processor.generate()
