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


from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.common.log import logger
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
    def create(api_name_with_id: str):
        perturbation = PerturbationFactory.perturbations.get(Config.pert_type)
        if perturbation:
            return perturbation(api_name_with_id)
        else:
            logger.error(f'{Config.pert_type} is a invalid perturbation type')
            raise ValueError
