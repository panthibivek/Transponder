from typing import Iterator, Optional
from torch.nn.parameter import Parameter
from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from utils.SamplingStrategies import SamplingStrategy
from utils.PonderingCriteria import PonderingCriteria
from typing import List


class Transponder(nn.Module):

    def __init__(
        self,
        llm_checkpoint: str,
        mlm_head: nn.Module,
        sampling_strategy: SamplingStrategy,
        pondering_criteria: PonderingCriteria,
        ponder_context_length: int,
    ):
        super().__init__()
        llm = AutoModelForCausalLM.from_pretrained(llm_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_checkpoint)
        self.backbone = llm.model
        self.lm_head = llm.lm_head
        self.mlm_head = mlm_head
        self.sampling_strategy = sampling_strategy
        self.pondering_criteria = pondering_criteria
        self.tokens = []
        self.attention_mask = []
        self.ponder_context_length = ponder_context_length
        self.logger = logging.getLogger("transponder_logger")

        self.logger.debug(f"EOS token id = {self.tokenizer.eos_token_id}")

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.mlm_head.parameters(recurse)

    def __append_token_to_state(self, next_token_logits: torch.Tensor):
        next_token = self.sampling_strategy.get_next_token_id(next_token_logits)
        self.tokens = torch.hstack([self.tokens, next_token])
        self.attention_mask = torch.hstack([self.attention_mask, torch.tensor([[1]])])

    def __revise_state(self, ponder_token_logits: torch.Tensor):
        # TODO: Different sampling strategies for lm and mlm?
        ponder_token = self.sampling_strategy.get_next_token_id(ponder_token_logits)
        self.tokens[0, -1 * (self.ponder_context_length + 1)] = ponder_token
        self.tokens = self.tokens[:, 0 : -1 * self.ponder_context_length]
        self.attention_mask = self.attention_mask[
            :, 0 : -1 * self.ponder_context_length
        ]

    def ponder_routine(self):
        self.logger.debug(
            f"Pondering on token generated at position {self.tokens.shape[-1]}"
        )
        # future context generation
        for i in range(self.ponder_context_length):
            backbone_output = self.backbone(
                input_ids=self.tokens, attention_mask=self.attention_mask
            )
            # TODO: Indexing -1 will not work in batches where tokens may be present
            # get the next token pos from the attention_mask above
            # i.e. the last index within each row that is a 1
            last_token_last_hidden_state = backbone_output.last_hidden_state[:, -1, :]
            lm_logits = self.lm_head(last_token_last_hidden_state)
            self.__append_token_to_state(lm_logits)
        # actual pondering
        self.attention_mask[0, -1 * (self.ponder_context_length + 1)] = (
            0  # so that the token being pondered about is invisible
        )
        self.logger.debug(f"Ponder head attn_mask = {self.attention_mask}")
        backbone_output = self.backbone(
            input_ids=self.tokens, attention_mask=self.attention_mask
        )
        last_token_last_hidden_state = backbone_output.last_hidden_state[:, -1, :]
        # TODO: Maybe discuss the possilibility of adding a positional encoding denoting the distance from the last token
        # where the token being pondered about is
        mlm_logits = self.mlm_head(last_token_last_hidden_state)
        self.__revise_state(mlm_logits)

    def __log_current_state(self):
        self.logger.debug(
            f"tokens = {self.tokens} | attn_mask = {self.attention_mask} | decoded_sentence = {self.tokenizer.batch_decode(self.tokens)}"
        )

    def generation_step(self):
        backbone_output = self.backbone(
            input_ids=self.tokens, attention_mask=self.attention_mask
        )
        # TODO: Indexing -1 will not work in batches where tokens may be present
        # get the next token pos from the attention_mask above
        # i.e. the last index within each row that is a 1
        last_token_last_hidden_state = backbone_output.last_hidden_state[:, -1, :]
        lm_logits = self.lm_head(last_token_last_hidden_state)
        self.__append_token_to_state(lm_logits)
        self.__log_current_state()
        if self.pondering_criteria.pondering_needed(
            dist=torch.nn.functional.softmax(lm_logits, dim=-1)
        ):
            self.ponder_routine()
            self.__log_current_state()

    def generate(self, prompt: str, num_new_tokens: Optional[int] = None):
        model_input = self.tokenizer(prompt, return_tensors="pt")
        self.tokens = model_input.input_ids
        self.attention_mask = model_input.attention_mask
        self.logger.debug(str(model_input))

        if num_new_tokens is not None:
            for _ in range(num_new_tokens):
                self.generation_step()
        else:
            while self.tokens[0, -1] != self.tokenizer.eos_token_id:
                self.generation_step()

        return self.tokenizer.batch_decode(self.tokens)

    @torch.no_grad()  # no gradients required for the hidden_states or anything producing them. Makes training efficient
    def get_last_token_last_hidden_state(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        last_hidden_state = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return last_hidden_state[:, -1, :]
