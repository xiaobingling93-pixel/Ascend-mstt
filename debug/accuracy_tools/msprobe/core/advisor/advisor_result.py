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

import os
import time

from msprobe.core.advisor.advisor_const import AdvisorConst
from msprobe.core.common.log import logger
from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.file_utils import change_mode, FileOpen


class AdvisorResult:
    """
    Class for generate advisor result
    """

    def __init__(self, node, line, message):
        self.suspect_node = node
        self.line = line
        self.advisor_message = message

    @staticmethod
    def gen_summary_file(out_path, message_list, suffix):
        file_name = 'advisor{}_{}.txt'.format(suffix, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
        result_file = os.path.join(out_path, file_name)
        try:
            with FileOpen(result_file, 'w+') as output_file:
                output_file.truncate(0)
                message_list = [message + AdvisorConst.NEW_LINE for message in message_list]
                output_file.writelines(message_list)
            change_mode(result_file, FileCheckConst.DATA_FILE_AUTHORITY)
        except IOError as io_error:
            logger.error("Failed to save %s, the reason is %s." % (result_file, io_error))
        else:
            logger.info("The advisor summary is saved in: %s" % result_file)

    def print_advisor_log(self):
        logger.info("The summary of the expert advice is as follows: ")
        message_list = [
            AdvisorConst.LINE + AdvisorConst.COLON + str(self.line),
            AdvisorConst.SUSPECT_NODES + AdvisorConst.COLON + self.suspect_node,
            AdvisorConst.ADVISOR_SUGGEST + AdvisorConst.COLON + self.advisor_message
        ]
        for message in message_list:
            logger.info(message)
        return message_list
