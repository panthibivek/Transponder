from .SamplingStrategy import SamplingStrategy
from torch import Tensor
from random import randint


class TopKSamplingStrategy(SamplingStrategy):

    def __init__(self, vocab_size: int, k: int):
        super(TopKSamplingStrategy, self).__init__(vocab_size)
        self.k = k

    def get_next_token_id(self, dist: Tensor) -> Tensor:
        super(TopKSamplingStrategy, self).get_next_token_id(dist)
        indices = dist.sort(dim=1).indices
        top_k_indices = indices[-1 * self.k :]
        random_k = randint(0, self.k - 1)
        return top_k_indices[:, random_k]
