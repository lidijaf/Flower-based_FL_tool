from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseAlgorithm(ABC):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def validate_config(self) -> None:
        return

    def initialize_state(self) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def fit(
        self,
        client: Any,
        parameters: List,
        config: Dict[str, Any],
    ):
        """
        Returns Flower-compatible fit tuple:
        (parameters, num_examples, metrics)
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        client: Any,
        parameters: List,
        config: Dict[str, Any],
    ):
        """
        Returns Flower-compatible evaluate tuple:
        (loss, num_examples, metrics)
        """
        pass
