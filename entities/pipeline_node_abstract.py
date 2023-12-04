from abc import ABC, abstractmethod


class PipelineNode(ABC):

    @abstractmethod
    def run(self):
        pass
