import unittest
from transponder import Transponder
from utils.SamplingStrategies import GreedySamplingStrategy, TopKSamplingStrategy
from utils.PonderingCriteria import EntropyPonderingCriteria
import torch
from torch import nn
import logging
import sys


class GreedyDecodingTest(unittest.TestCase):

    def testThatItRuns(self):
        SamplingStrategy = GreedySamplingStrategy(32000)
        token_ids = [0]
        for _ in range(100):
            distribution = torch.randn(1, 32000)
            next_token_id = SamplingStrategy.get_next_token_id(distribution)
            token_ids.append(next_token_id)


class TopKDecodingTest(unittest.TestCase):

    def testThatItRuns(self):
        token_ids = [0]
        for _ in range(100):
            SamplingStrategy = TopKSamplingStrategy(10, 3)
            dist = torch.randint(1, 10, (1, 10))
            next_token_id = SamplingStrategy.get_next_token_id(dist)
            token_ids.append(next_token_id)


class TestTransponder(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger("transponder_logger")
        super(TestTransponder, self).__init__(*args, **kwargs)

    def testTransponderWorks(self):
        LLM_VOCAB_SIZE = 32000
        LLM_HIDDEN_SIZE = 2048

        mlm_head = nn.Sequential(
            *[
                nn.Linear(LLM_HIDDEN_SIZE, 512, bias=True),
                nn.Linear(512, LLM_VOCAB_SIZE),
            ]
        )

        transponder = Transponder(
            llm_checkpoint="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            mlm_head=mlm_head,
            sampling_strategy=GreedySamplingStrategy(LLM_VOCAB_SIZE),
            pondering_criteria=EntropyPonderingCriteria(
                threshold=8, vocab_size=LLM_VOCAB_SIZE
            ),
            ponder_context_length=5,
        )

        generated_sentence = transponder.generate("I love jello, and", 10)
        logger = logging.getLogger("transponder_logger")
        logger.debug(f"Generated_sentence = {generated_sentence}")

    def test_last_token_last_hidden_state(self):
        LLM_VOCAB_SIZE = 32000
        LLM_HIDDEN_SIZE = 2048

        mlm_head = nn.Sequential(
            *[
                nn.Linear(LLM_HIDDEN_SIZE, 512, bias=True),
                nn.Linear(512, LLM_VOCAB_SIZE),
            ]
        )

        transponder = Transponder(
            llm_checkpoint="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            mlm_head=mlm_head,
            sampling_strategy=GreedySamplingStrategy(LLM_VOCAB_SIZE),
            pondering_criteria=EntropyPonderingCriteria(
                threshold=3, vocab_size=LLM_VOCAB_SIZE
            ),
            ponder_context_length=5,
        )

        backbone_inputs = transponder.tokenizer(["I am Iron Man."], return_tensors="pt")
        self.logger.debug(f"backbone_inputs = {backbone_inputs}")
        last_token_last_hidden_state = transponder.get_last_token_last_hidden_state(
            **backbone_inputs
        )
        self.logger.debug(
            f"last_token_last_hidden_state = {last_token_last_hidden_state.shape}"
        )


if __name__ == "__main__":
    with open("test_log.log", "w") as log_file:
        logging.basicConfig(stream=log_file)
        logging.getLogger("transponder_logger").setLevel(logging.DEBUG)
        unittest.main()
