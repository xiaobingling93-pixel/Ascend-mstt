from abc import ABC, abstractmethod
from typing import Any

from msprobe.pytorch.free_benchmark.common.params import DataParams


class BaseLayer(ABC):
    def __init__(self, api_name: str) -> None:
        self.api_name = api_name

    @abstractmethod
    def handle(self, params: DataParams) -> Any:
        pass
