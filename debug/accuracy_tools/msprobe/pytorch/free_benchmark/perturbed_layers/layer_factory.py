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


from msprobe.pytorch.free_benchmark import FreeBenchmarkException
from msprobe.pytorch.free_benchmark.common.enums import DeviceType, PerturbationMode
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.add_noise import AddNoiseLayer
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.bit_noise import BitNoiseLayer
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.change_value import (
    ChangeValueLayer,
)
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.improve_precision import (
    ImprovePrecisionLayer,
)
from msprobe.pytorch.free_benchmark.perturbed_layers.npu.no_change import NoChangeLayer
from msprobe.pytorch.free_benchmark.perturbed_layers.run_cpu import CpuLayer


class LayerFactory:
    layers = {
        DeviceType.NPU: {
            PerturbationMode.ADD_NOISE: AddNoiseLayer,
            PerturbationMode.CHANGE_VALUE: ChangeValueLayer,
            PerturbationMode.NO_CHANGE: NoChangeLayer,
            PerturbationMode.BIT_NOISE: BitNoiseLayer,
            PerturbationMode.IMPROVE_PRECISION: ImprovePrecisionLayer,
        },
        DeviceType.CPU: {PerturbationMode.TO_CPU: CpuLayer},
    }

    @staticmethod
    def create(api_name: str, device_type: str, mode: str):
        layer = LayerFactory.layers.get(device_type)
        if not layer:
            raise FreeBenchmarkException(
                FreeBenchmarkException.UnsupportedType,
                f"无标杆工具不支持当前设备 {device_type}",
            )
        layer = layer.get(mode)
        if not layer:
            raise FreeBenchmarkException(
                FreeBenchmarkException.UnsupportedType,
                f"无标杆工具无法识别该扰动因子 {mode} on {device_type}",
            )
        return layer(api_name)
