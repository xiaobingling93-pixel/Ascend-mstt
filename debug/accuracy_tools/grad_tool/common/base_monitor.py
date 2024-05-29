from abc import ABC, abstractmethod

from grad_tool.common.utils import get_config


class BaseMonitor(ABC):

    def __init__(self, config_file):
        self.config = get_config(config_file)

    @abstractmethod
    def monitor(self, module):
        raise NotImplementedError("monitor is not implemented.")
