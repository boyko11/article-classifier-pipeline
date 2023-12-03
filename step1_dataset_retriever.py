import nltk

from pipeline_node_abstract import PipelineNode
from torchtext.datasets import AG_NEWS


class DatasetRetrieverReuters(PipelineNode):

    def run(self):
        nltk.download('reuters')


class DatasetRetrieverAGNews(PipelineNode):

    def run(self):
        train_iter, test_iter = AG_NEWS()
        return train_iter, test_iter
