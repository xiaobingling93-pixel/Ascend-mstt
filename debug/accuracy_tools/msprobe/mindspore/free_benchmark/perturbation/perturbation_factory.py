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

from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.free_benchmark.perturbation.add_noise import AddNoisePerturbation
from msprobe.mindspore.free_benchmark.perturbation.bit_noise import BitNoisePerturbation
from msprobe.mindspore.free_benchmark.perturbation.exchange_value import ExchangeValuePerturbation
from msprobe.mindspore.free_benchmark.perturbation.improve_precision import ImprovePrecisionPerturbation
from msprobe.mindspore.free_benchmark.perturbation.no_change import NoChangePerturbation


class PerturbationFactory:
    """
    扰动工厂类

    """
    perturbations = {
        FreeBenchmarkConst.IMPROVE_PRECISION: ImprovePrecisionPerturbation,
        FreeBenchmarkConst.ADD_NOISE: AddNoisePerturbation,
        FreeBenchmarkConst.BIT_NOISE: BitNoisePerturbation,
        FreeBenchmarkConst.NO_CHANGE: NoChangePerturbation,
        FreeBenchmarkConst.EXCHANGE_VALUE: ExchangeValuePerturbation
    }

    @staticmethod
    def create(api_name: str):
        perturbation = PerturbationFactory.perturbations.get(Config.pert_type)
        if perturbation:
            return perturbation(api_name)
        else:
            raise Exception(f'{Config.pert_type} is a invalid perturbation type')
