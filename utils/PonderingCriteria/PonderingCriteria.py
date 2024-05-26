from abc import ABC, abstractmethod
from torch import Tensor


class PonderingCriteria(ABC):

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    @abstractmethod
    def pondering_needed(self, dist: Tensor) -> bool:
        pass
