import statistics  as st
from abc import ABC
from typing import List
import sys
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

class ScanRule(ABC):
    def apply(self, history, cur):
        raise NotImplemented("abstract method apply is not implemented")

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
            rule_cls_name = spec["rule_name"]
            rule_args = spec["args"]
            cur_module = sys.modules[__name__]
            rule_cls = getattr(cur_module, rule_cls_name)
            rule_instance = rule_cls(**rule_args)
            alert_rules.append(rule_instance)
        return alert_rules

    @staticmethod
    def scan(scan_rules: List[ScanRule], history, cur):
        anomaly = False
        for rule in scan_rules:
            anomaly = rule.apply(history, cur)
            if anomaly:
                return anomaly, rule.name
        return anomaly, None

class bcolors:
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
    def __init__(self, path, ad_rules, anomaly_inform=False):
        super().__init__(path)
        self.tag2scalars = defaultdict(list)
        self.ad_rules = ad_rules
        self.anomaly_inform = anomaly_inform
    
    def _ad(self, scalar_value, history):
        return AnomalyScanner.scan(self.ad_rules, history, cur=scalar_value)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False, double_precision=False):
        new_avg = avg = scalar_value
        if tag in self.tag2scalars:
            N = len(self.tag2scalars[tag])
            _, avg = self.tag2scalars[tag][-1]
            new_avg = (avg*N + scalar_value)/(N + 1)
        self.tag2scalars[tag].append((scalar_value, new_avg))    
        detected, rule_name = self._ad(scalar_value, history=avg)
        if detected:
            print(f"{bcolors.WARNING}> Rule {rule_name} reports anomaly signal in {tag} at step {global_step}.{bcolors.ENDC}")
            exception_message = f"{bcolors.WARNING}> Rule {rule_name} reports anomaly signal in {tag} at step {global_step}.{bcolors.ENDC}"
            if self.anomaly_inform:
                self.anomaly_inform.run(exception_message)  
        return super().add_scalar(tag, scalar_value, global_step, walltime, new_style, double_precision)
