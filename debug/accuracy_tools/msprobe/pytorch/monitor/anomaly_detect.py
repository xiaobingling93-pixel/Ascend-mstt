#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import os
import sys
import statistics as st
from abc import ABC
from typing import List
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import check_path_before_create, change_mode, create_directory
from msprobe.core.common.const import FileCheckConst


class ScanRule(ABC):
    def apply(self, history, cur):
        raise NotImplementedError("abstract method apply is not implemented")


class AnomalyTurbulence(ScanRule):
    name = "AnomalyTurbulence"

    def __init__(self, threshold) -> None:
        self.threshold = threshold

    def apply(self, history, cur):
        baseline = st.mean(history) if isinstance(history, list) else history

        up_bound = baseline + baseline * self.threshold
        if baseline > 0:
            return cur > up_bound
        else:
            return cur < up_bound


class AnomalyScanner:

    @staticmethod
    def load_rules(specs: List[dict]):
        if specs is None:
            return []
        alert_rules = []
        for spec in specs:
            # 使用get方法获取键值，如果键不存在则返回None
            rule_cls_name = spec.get("rule_name")
            rule_args = spec.get("args")

            # 检查必要的键是否存在
            if rule_cls_name is None or rule_args is None:
                logger.warning(f"Spec is missing required keys: {spec}")
                continue

            cur_module = sys.modules[__name__]
            try:
                rule_cls = getattr(cur_module, rule_cls_name)
            except AttributeError:
                logger.error(f"Rule class '{rule_cls_name}' not found in the current module.")
                continue

            try:
                rule_instance = rule_cls(**rule_args)
                alert_rules.append(rule_instance)
            except Exception as e:
                logger.error(f"Error creating instance of rule '{rule_cls_name}': {e}")
                continue

        return alert_rules

    @staticmethod
    def scan(scan_rules: List[ScanRule], history, cur):
        anomaly = False
        for rule in scan_rules:
            anomaly = rule.apply(history, cur)
            if anomaly:
                return anomaly, rule.name
        return anomaly, None


class BCOLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SummaryWriterWithAD(SummaryWriter):
    def __init__(self, path, ad_rules, job_id, anomaly_inform=False):
        check_path_before_create(path)
        create_directory(path)
        try:
            super().__init__(path)
        except Exception as e:
            logger.error(f'error when init summary writer at {path}: {e}')
            raise ValueError("Init summary writer error.") from e
        for event in os.listdir(path):
            change_mode(os.path.join(path, event), FileCheckConst.DATA_FILE_AUTHORITY)
        self.tag2scalars = defaultdict(list)
        self.ad_rules = ad_rules
        self.job_id = job_id
        self.anomaly_inform = anomaly_inform

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False):
        new_avg = avg = scalar_value
        if tag in self.tag2scalars:
            n = len(self.tag2scalars[tag])
            _, avg = self.tag2scalars[tag][-1]
            new_avg = (avg * n + scalar_value) / (n + 1)
        self.tag2scalars[tag].append((scalar_value, new_avg))
        detected, rule_name = self._ad(scalar_value, history=avg)
        if detected:
            logger.info(
                f"{BCOLORS.WARNING}> Rule {rule_name} reports anomaly signal in {tag} at step {global_step}."
                f"{BCOLORS.ENDC}")
            exception_message = (f"{BCOLORS.WARNING}> Rule {rule_name} reports anomaly signal in {tag} at step "
                                 f"{global_step}.{BCOLORS.ENDC}")
            if self.anomaly_inform:
                self.anomaly_inform.run(exception_message, self.job_id)
        args = [tag, scalar_value, global_step, walltime, new_style, double_precision]
        return super().add_scalar(*args)

    def _ad(self, scalar_value, history):
        return AnomalyScanner.scan(self.ad_rules, history, cur=scalar_value)
