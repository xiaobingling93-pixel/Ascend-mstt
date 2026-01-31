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


from msprobe.core.common.const import Const


class PerturbationMode:
    ADD_NOISE = "add_noise"
    CHANGE_VALUE = "change_value"
    IMPROVE_PRECISION = "improve_precision"
    NO_CHANGE = "no_change"
    BIT_NOISE = "bit_noise"
    TO_CPU = "to_cpu"


class DeviceType:
    NPU = "npu"
    CPU = "cpu"


class FuzzThreshold:
    BF16_THD = 1e-4
    F16_THD = 1e-6
    F32_THD = 1e-8
    F64_THD = 1e-16


class NormType:
    ONE_NORM = (1, "one_norm")
    TWO_NORM = (2, "two_norm")
    ENDLESS_NORM = (3, "endless_norm")


class HandlerType:
    CHECK = "check"
    PREHEAT = "preheat"
    FIX = "fix"


class FuzzLevel:
    BASE_LEVEL = "L1"
    ADV_LEVEL = "L2"
    REAL_LEVEL = "L3"


class PytorchFreeBenchmarkConst:
    PERTURBATION_MODE_LIST = [
        PerturbationMode.ADD_NOISE,
        PerturbationMode.CHANGE_VALUE,
        PerturbationMode.IMPROVE_PRECISION,
        PerturbationMode.NO_CHANGE,
        PerturbationMode.BIT_NOISE,
        PerturbationMode.TO_CPU,
    ]
    DEFAULT_MODE = PerturbationMode.IMPROVE_PRECISION
    DEVICE_LIST = [DeviceType.NPU, DeviceType.CPU]
    DEFAULT_DEVICE = DeviceType.NPU
    HANDLER_LIST = [HandlerType.CHECK, HandlerType.FIX]
    DEFAULT_HANDLER = HandlerType.CHECK
    FUZZ_LEVEL_LIST = [FuzzLevel.BASE_LEVEL]
    DEFAULT_FUZZ_LEVEL = FuzzLevel.BASE_LEVEL
    FUZZ_STAGE_LIST = [Const.FORWARD, Const.BACKWARD]
    FIX_MODE_LIST = [PerturbationMode.IMPROVE_PRECISION, PerturbationMode.TO_CPU]
    DEFAULT_FUZZ_STAGE = Const.FORWARD
    DEFAULT_PREHEAT_STEP = 15
    DEFAULT_MAX_SAMPLE = 20
    CPU_MODE_LIST = [PerturbationMode.TO_CPU]
    FIX_STAGE_LIST = [Const.FORWARD]
