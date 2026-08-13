"""Shared base class for distributed trainers."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTrainer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def setup_distributed(self) -> None:
        ...

    @abstractmethod
    def setup_model(self) -> None:
        ...

    @abstractmethod
    def train(self) -> None:
        ...
