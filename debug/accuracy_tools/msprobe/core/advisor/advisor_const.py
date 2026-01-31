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


class AdvisorConst:
    """
    Class for advisor const
    """

    # text symbol
    NEW_LINE = "\n"
    COLON = ": "

    # advisor summary key
    SUSPECT_NODES = "Suspect Nodes"
    LINE = "Line"
    ADVISOR_SUGGEST = "Expert Advice"

    NO_ERROR_API = "NA"

    # advisor message
    NO_ERR_SUGGEST = "All data in comparison result meets the accuracy requirements."
    FORWARD_INPUT_SUGGEST = "1. Analyze the model to view the input source.\n" \
                            "2. Check whether an inplace API causes the output result to overwrite the input result. "\
                            "That is, the fault is actually caused by a computation error.\n" \
                            "3. The fault may be caused by memory corruption and further analysis is required."
    FORWARD_OUTPUT_SUGGEST = "This is a forward API computation error. Check the computation implementation."
    BACKWARD_INPUT_SUGGEST = "Check whether the forward computation result is affected."
    BACKWARD_OUTPUT_SUGGEST = "This is a backward API computation error. Check the computation implementation."
    BATCH_NORM_SUGGEST = "Torch API batch_norm input not fixed, the following suggestions may fix it:\n" \
                         "1. If use torch.nn.functional.batch_norm, you can set parameter training=False.\n" \
                         "2. If use torch.nn.BatchNormXXX, you can set parameter affine=False.\n" \
                         "3. Use seed_all(mode=True) to enable deterministic computing."
    DETERMINISTIC_SUGGEST = "This torch api may be uncertainty in the calculation, " \
                            "can seed_all(mode=True) to enable deterministic computing."

    FUNC_BATCH_NORM = "Functional_batch_norm"
    FORWARD_INPUT_1 = "forward_input.1"
    NEED_DETERMINISTIC_API = ["conv2d", "conv3d", "matmul", "nll_loss", "layer_norm", "lstm"]
    BATCH_NORM = "batch_norm"

    # name keyword
    INPUT = "input"
    OUTPUT = "output"
    FORWARD = "forward"
    BACKWARD = "backward"
