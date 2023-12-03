import numpy as np
import pandas as pd

from pipeline_node_abstract import PipelineNode
from util.plot_util import plot_num_docs_per_category_chart, plot_doc_len_distribution_chart


class Eda(PipelineNode):

    def __init__(self):
        self.data = None

    def _create_dataframe(self, dataset_iters):
        data = {
            "label": [],
            "text": [],
            "num_words": []
        }
        for dataset_iter in dataset_iters:
            for label, text in dataset_iter:
                data["label"].append(label)
                words = text.split()
                data["text"].append(text)
                data["num_words"].append(len(words))
        return pd.DataFrame(data)

    def print_stats(self):

        print(self.data.describe())
        print('------------------')

        print("Total Documents:", len(self.data))
        print('------------------')

        print("Sample Documents:")
        print(self.data.head())
        print('------------------')

        print("Sample Texts:")
        # sample 10 random texts
        sample_texts = self.data['text'][np.random.randint(0, self.data.shape[0], 10)]
        for sample_text in sample_texts:
            print(sample_text)
            print('------------------')

        print('------------------')

    def run(self, train_iter, test_iter):
        combined_iter = [train_iter, test_iter]
        self.data = self._create_dataframe(combined_iter)
        self.print_stats()
        plot_num_docs_per_category_chart(self.data['label'].value_counts())
        plot_doc_len_distribution_chart(self.data['num_words'])

        return train_iter, test_iter
