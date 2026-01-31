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


import unittest
from unittest.mock import patch

from msprobe.mindspore.common.log import logger
from msprobe.mindspore.free_benchmark.perturbation.perturbation_factory import PerturbationFactory
from msprobe.mindspore.free_benchmark.common.config import Config
from msprobe.mindspore.common.const import FreeBenchmarkConst
from msprobe.mindspore.free_benchmark.perturbation.add_noise import AddNoisePerturbation
from msprobe.mindspore.free_benchmark.perturbation.bit_noise import BitNoisePerturbation
from msprobe.mindspore.free_benchmark.perturbation.no_change import NoChangePerturbation
from msprobe.mindspore.free_benchmark.perturbation.improve_precision import ImprovePrecisionPerturbation
from msprobe.mindspore.free_benchmark.perturbation.exchange_value import ExchangeValuePerturbation


class TestPerturbationFactory(unittest.TestCase):

    @patch.object(logger, "error")
    def test_create(self, mock_logger_error):
        api_name = "Functional.add.0"

        Config.pert_type = "UNKNOWN"
        with self.assertRaises(ValueError):
            PerturbationFactory.create(api_name)
        mock_logger_error.assert_called_with("UNKNOWN is a invalid perturbation type")

        Config.pert_type = FreeBenchmarkConst.EXCHANGE_VALUE
        pert = PerturbationFactory.create(api_name)
        self.assertTrue(isinstance(pert, ExchangeValuePerturbation))
        self.assertEqual(pert.api_name_with_id, api_name)
        self.assertFalse(pert.is_fuzzed)

        Config.pert_type = FreeBenchmarkConst.NO_CHANGE
        pert = PerturbationFactory.create(api_name)
        self.assertTrue(isinstance(pert, NoChangePerturbation))
        self.assertEqual(pert.api_name_with_id, api_name)
        self.assertFalse(pert.is_fuzzed)

        Config.pert_type = FreeBenchmarkConst.BIT_NOISE
        pert = PerturbationFactory.create(api_name)
        self.assertTrue(isinstance(pert, BitNoisePerturbation))
        self.assertEqual(pert.api_name_with_id, api_name)
        self.assertFalse(pert.is_fuzzed)

        Config.pert_type = FreeBenchmarkConst.ADD_NOISE
        pert = PerturbationFactory.create(api_name)
        self.assertTrue(isinstance(pert, AddNoisePerturbation))
        self.assertEqual(pert.api_name_with_id, api_name)
        self.assertFalse(pert.is_fuzzed)

        Config.pert_type = FreeBenchmarkConst.IMPROVE_PRECISION
        pert = PerturbationFactory.create(api_name)
        self.assertTrue(isinstance(pert, ImprovePrecisionPerturbation))
        self.assertEqual(pert.api_name_with_id, api_name)
        self.assertFalse(pert.is_fuzzed)
