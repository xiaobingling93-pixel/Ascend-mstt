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

class OperatorPrompt(object):
    RANK_ID = "{}号卡"
    PYTORCH_OPERATOR_TUNE_SUGGESTION = "通过AOE优化算子，使用样例如下：\n" \
                                        "'aoe --job_type=2 --model_path=$user_dump_path " \
                                        "--tune_ops_file={}'\n"
    MSLITE_OPERATOR_TUNE_SUGGESTION = f"在Mindpore Lite 框架通过AOE优化算子，使用样例如下：\n" \
                                      f"converter_lite --fmk=ONNX --optimize=ascend_oriented --saveType=MINDIR " \
                                      f"--modelFile=$user_model.onnx --outputFile=user_model " \
                                      f"--configFile=./config.txt\n"
    PYTORCH_RELEASE_SUGGESTION = "详细信息请参考：<a href={} target='_blank'>链接</a>"
    MSLITE_RELEASE_SUGGESTION = "\nMSLite AOE的配置文件如下usage：\n" \
                                "[ascend_context]\n" \
                                "aoe_mode=\"operator tuning\"\n" \
                                "--tune_ops_file={}\n" \
                                "\n详细信息请参考：<a href=" \
                                "{} target='_blank'>链接</a>"