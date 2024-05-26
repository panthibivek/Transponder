from .SamplingStrategy import SamplingStrategy
from torch import Tensor


class GreedySamplingStrategy(SamplingStrategy):

    def get_next_token_id(self, dist: Tensor) -> Tensor:
        super(GreedySamplingStrategy, self).get_next_token_id(dist)
        return dist.argmax(1).reshape(1, 1)
