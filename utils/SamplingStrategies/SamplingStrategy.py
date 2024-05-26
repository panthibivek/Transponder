from abc import ABC
from torch import Tensor


class SamplingStrategy(ABC):

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def get_next_token_id(self, dist: Tensor) -> Tensor:
        assert (
            dist.shape[0] == 1 and dist.shape[1] == self.vocab_size
        ), f"Distribution shape not valid. Expected distribution size =  (1, {self.vocab_size}), but got a tensor of size {dist.shape}"
