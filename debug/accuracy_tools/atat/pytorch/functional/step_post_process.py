from abc import ABC, abstractmethod
from ..common.exceptions import StepException


def run_parallel_ut(config):
    pass


def compare_distrbuted(config):
    pass


def build_step_post_process(config):
    if not config.on_step_end:
        return None
    if config.on_step_end == StepPostProcess.SingleAPICheck:
        return SingleAPICheck(config)
    elif config.on_step_end == StepPostProcess.Compare:
        return AutoCompare(config)
    else:
        raise StepException(StepException.InvalidPostProcess, f"step后处理须配置为"
            f"'{StepPostProcess.SingleAPICheck}'或'{StepPostProcess.Compare}'，"
            f"实际配置为{config.on_step_end}")


class StepPostProcess(ABC):
    SingleAPICheck = 'single_api_check'
    Compare = 'compare'


class SingleAPICheck:
    def __init__(self, config):
        self.config = config

    def run(self):
        run_parallel_ut(self.config)

class AutoCompare:
    def __init__(self, config):
        self.config = config

    def run(self):
        compare_distrbuted(self.config.bench_dump_path, self.config.dump_path)
