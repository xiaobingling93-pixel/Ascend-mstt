# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
