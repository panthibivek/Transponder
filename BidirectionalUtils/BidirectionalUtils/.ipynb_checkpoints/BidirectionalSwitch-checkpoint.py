from abc import ABC, abstractmethod

class BidirectionalSwitch(ABC):

    def __init__(self, llm):
        self.llm = llm

    @abstractmethod
    def __enter__(self):
        # goal change the state in the transformer decoder layers of the LLM
        # to induce a bidirectional behaviour
        pass

    @abstractmethod
    def __exit__(self, type, value, traceback):
        # restore the transformer decoder layers of the LLM to their original state
        pass