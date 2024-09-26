# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
