# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import itertools
import os
import sys
import statistics as st
from abc import ABC
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict

import pandas as pd

from mindspore import ops
from mindspore import _no_grad
from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import change_mode, create_directory, write_df_to_csv
from msprobe.core.common.const import FileCheckConst, MonitorConst


class ScanRule(ABC):
    name = "ScanRule"

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
        """
        specs: [{"rule_name": "AnomalyTurbulence", "args": {"threshold": 0.5}}]
        """
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

            cur_module = sys.modules.get(__name__)
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


class AnomalyDataFactory(ABC):
    def __init__(self, rank, pp_stage, group_mates):
        super().__init__()
        self.rank = rank
        self.pp_stage = pp_stage
        self.group_mates = group_mates
        self.micro_step = 0
        self.name2callid = {}

    def set_call_id(self, name2callid):
        """根据当前GradContext信息更新call_id vpp_stage等信息
        """
        self.name2callid = name2callid

    def create(self, tag, message, step):
        """如果检查出异常, 调用当前接口生成GradAnomalyData实例
        tag (tuple): metric tag ('0:1.post_attention_norm.weight/rank0/pre_grad', 'min')
        message (str): anomaly detect message
        step (int): training step
        """
        if not isinstance(tag, tuple) or len(tag) != 2:
            raise ValueError("tag must be a tuple with length 2")
        tag_name = tag[0]
        param_name = tag_name.split('/')[0]
        call_id = self.name2callid.get(tag_name, -1)
        if MonitorConst.NAME_SEP in param_name:
            vpp_stage = int(param_name.split(MonitorConst.NAME_SEP)[0])
        else:
            vpp_stage = 0

        return GradAnomalyData(
            self.rank,
            step,
            self.micro_step,
            self.pp_stage,
            vpp_stage,
            call_id,
            tag_name,
            message,
            self.group_mates
        )


class TrainStage:
    DEFAULT_STAGE = -1
    FORWARD_STAGE = 0
    BACKWARD_STAGE = 1
    OPTIMIZER_STAGE = 2


FORWARD_KEY = [MonitorConst.ACTV_IN, MonitorConst.ACTV_OUT]
BACKWARD_KEY = [MonitorConst.ACTVGRAD_IN, MonitorConst.ACTVGRAD_OUT,
                MonitorConst.PRE_GRAD, MonitorConst.POST_GRAD, MonitorConst.ACC_GRAD]
OPTIMIZER_KEY = [MonitorConst.EXP_AVG, MonitorConst.EXP_AVG_SQ]
TRAIN_STAGE = {
    **{key_: TrainStage.FORWARD_STAGE for key_ in FORWARD_KEY},
    **{key_: TrainStage.BACKWARD_STAGE for key_ in BACKWARD_KEY},
    **{key_: TrainStage.OPTIMIZER_STAGE for key_ in OPTIMIZER_KEY}
}


@dataclass(eq=True)
class GradAnomalyData:
    rank: int = 0
    step: int = 0
    micro_step: int = 0
    pp_stage: int = 0
    vpp_stage: int = 0
    call_id: int = 0
    tag_name: str = field(default=None, compare=False)
    message: str = field(default="", compare=False)
    group_mates: list = field(default=None, compare=False)

    def __lt__(self, other):
        """
        自定义比较函数，用于确定 GradAnomalyData 实例之间的顺序。
        比较规则为：
            step 和 micro_step 值越小优先级越高；
            vpp 和 pp 在前向阶段值越小优先级越高，在非前向阶段值越大优先级越高；
            call_id 值越小优先级越高。
        """
        if not isinstance(other, GradAnomalyData):
            return NotImplemented

        self_train_stage = self.get_train_stage(self.tag_name)
        other_train_stage = self.get_train_stage(other.tag_name)

        def vpp_pp_comparator(anomaly):
            """
            Determine the priority rule for vpp and pp based on train stage
            Forward stage prefers smaller vpp and pp
            Other stages prefer larger vpp and pp
            """
            if self_train_stage == TrainStage.FORWARD_STAGE:
                return anomaly.vpp_stage, anomaly.pp_stage
            else:
                return -anomaly.vpp_stage, -anomaly.pp_stage

        self_cmp = [self.step, self.micro_step, self_train_stage, *vpp_pp_comparator(self), self.call_id]
        other_cmp = [other.step, other.micro_step, other_train_stage, *vpp_pp_comparator(other), other.call_id]
        return self_cmp < other_cmp

    def __le__(self, other):
        if not isinstance(other, GradAnomalyData):
            return NotImplemented
        return self == other or self < other

    @staticmethod
    def get_train_stage(tag_name):
        """
        :param tag_name: "0:fc2_0/rank0/input", "0:fc1.weight/rank0/post_grad", "0:fc2.weight/rank0/exp_avg_sq"
        :return: int, if forward return 0; if backward return 1; if optimizer return 2
        """
        key_ = tag_name.split("/")[-1]
        return TRAIN_STAGE.get(key_, TrainStage.DEFAULT_STAGE)

    def to_dict(self):
        return self.__dict__

    def get_key(self):
        # 0:1.self_attention.core_attention_flash_0/rank0/input_grad
        return ''.join([str(self.tag_name), "_step_", str(self.step), "_call_", str(self.call_id)])


@dataclass
class WriterInput:
    path: str
    ad_rules: list
    job_id: str
    anomaly_factory: AnomalyDataFactory = None
    ndigits: int = 6
    step_count_per_record: int = 1


class BaseWriterWithAD:
    def __init__(self, writer_input: WriterInput):
        self.tag2scalars = {}
        self.ad_rules = writer_input.ad_rules
        self.job_id = writer_input.job_id
        self.anomaly_factory = writer_input.anomaly_factory
        self.anomalies = []
        self.ndigits = writer_input.ndigits

    def get_anomalies(self):
        """返回已检测到的异常列表
        """
        return self.anomalies

    def clear_anomalies(self):
        self.anomalies.clear()

    def add_scalar(self, tag, scalar_value, global_step=None, need_explain=False):
        """If an anomaly is detected, the anomaly information is recorded and added to self.anomalies.
        Args:
            tag (tuple): tuple of tag_name and tag like ('0:1.post_attention_norm.weight/rank0/pre_grad', 'min').
            scalar_value (float): scalar_value.
            global_step (int): global_step.
        Returns:
            None
        """
        detected = False
        if self.ad_rules:
            avg = self._update_tag2scalars(tag, scalar_value)
            detected, rule_name = self._ad(scalar_value, history=avg)
        if detected:
            exception_message = f"Rule {rule_name} reports anomaly signal in {tag} at step {global_step}."
            logger.info(f"{BCOLORS.WARNING}> {exception_message}{BCOLORS.ENDC}")
            # append to self.anomalies for dump
            if self.anomaly_factory:
                self.anomalies.append(self.anomaly_factory.create(tag, exception_message, global_step))

    def write_metrics(self, op_list, metric_value, step, prefix='', need_explain=False):
        if not metric_value:
            return
        tensors = []
        tags = list(itertools.product(metric_value.keys(), op_list))
        for op2tensor in metric_value.values():
            tensors.extend(op2tensor.values())
        with _no_grad():
            metric_list = ops.stack(tensors).tolist() if tensors else []
        for tag, metric in zip(tags, metric_list):
            self.add_scalar(tag, metric, step, need_explain)

    def _ad(self, scalar_value, history):
        return AnomalyScanner.scan(self.ad_rules, history, cur=scalar_value)

    def _update_tag2scalars(self, tag, scalar_value):
        """Update the average and count of a scalar value associated with a tag.

        This method is used to maintain a running average of scalar values for each tag.


        Args:
            tag (str): The tag identifier.
            scalar_value (float): The scalar value to be added.

        Returns:
            float: The average value before update.
        """
        if tag not in self.tag2scalars:
            self.tag2scalars[tag] = {'avg': scalar_value, 'count': 0}
        avg = self.tag2scalars[tag]['avg']
        new_avg = (avg * self.tag2scalars[tag]['count'] + scalar_value) / (self.tag2scalars[tag]['count'] + 1)
        self.tag2scalars[tag]['avg'] = new_avg
        self.tag2scalars[tag]['count'] += 1
        return avg


class CSVWriterWithAD(BaseWriterWithAD):
    def __init__(self, writer_input: WriterInput):
        super().__init__(writer_input)

        path = writer_input.path
        self.log_dir = path
        create_directory(path)
        change_mode(path, FileCheckConst.DATA_DIR_AUTHORITY)
        self.context_dict = defaultdict(list)
        self.header = []
        self.step_count_per_record = writer_input.step_count_per_record

    def get_step_interval(self, step):
        count = step // self.step_count_per_record
        return count * self.step_count_per_record, (count + 1) * self.step_count_per_record - 1

    def write_csv(self, prefix, step):
        """
        Args:
            prefix[str]: prefix of output csv file e.g. grad_unreduced
            step[int]
        """
        if len(self.context_dict) == 0:
            return

        ster_start, step_end = self.get_step_interval(step)
        filepath = os.path.join(self.log_dir, f'{prefix}_{ster_start}-{step_end}.csv')
        if not os.path.exists(filepath):
            data_frame = pd.DataFrame(columns=self.header)
            write_df_to_csv(data_frame, filepath)

        new_data = []
        for name, metric_value in self.context_dict.items():
            if MonitorConst.NAME_SEP not in name:
                new_data.append([name] + [step] + metric_value)
            else:
                new_data.append(name.split(MonitorConst.NAME_SEP) + [step] + metric_value)
        new_data = pd.DataFrame(new_data).round(self.ndigits)
        write_df_to_csv(new_data, filepath, mode='a+', header=False)
        self.context_dict = defaultdict(list)

    def add_scalar(self, tag, scalar_value, global_step, need_explain=False):
        """
        ('0:1.post_attention_norm.weight/rank0/pre_grad', 'min')
        """
        super().add_scalar(tag, scalar_value, global_step, need_explain=False)
        split_name = tag[0].split('/')
        name = split_name[0]
        if need_explain:
            if 'pre' in split_name[-1]:
                name += '.input'
            if 'post' in split_name[-1]:
                name += '.output'
        self.context_dict[name].append(scalar_value)

    def write_metrics(self, op_list, metric_value, step, prefix='', need_explain=False):
        need_explain = prefix == 'other'
        super().write_metrics(op_list, metric_value, step, prefix='', need_explain=need_explain)

        # generate csv headers
        # set hashmap to reduce the number of headers generated.
        # 前向的norm用input.ops_和output.ops_，反向的用input_grad.ops_和output_grad.ops_
        if prefix in {"actv", "actv_grad"}:
            if prefix == "actv":
                input_and_output = [MonitorConst.ACTV_IN, MonitorConst.ACTV_OUT]
            else:
                input_and_output = [MonitorConst.ACTVGRAD_IN, MonitorConst.ACTVGRAD_OUT]
            ops_ = [MonitorConst.DOT.join(i) for i in itertools.product(input_and_output, op_list)]
            csv_header = ["module_name", "step", *ops_]
        else:
            csv_header = ["param_name", "step", *op_list]

        keys = list(metric_value.keys())
        if keys and MonitorConst.NAME_SEP in keys[0]:
            csv_header.insert(0, "vpp_stage")

        self.header = csv_header
        self.write_csv(prefix, step)
        self.header = []

    def close(self):
        pass
