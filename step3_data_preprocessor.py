import os
from typing import Tuple, List

import torch
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer

from pipeline_node_abstract import PipelineNode


class DataPreprocessor(PipelineNode):

    def __init__(self, embedding_dim=100, max_length=500, num_classes=4):
        self.tokenizer = get_tokenizer('basic_english')
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.vocab = GloVe(name='6B', dim=self.embedding_dim)
        self.num_classes = num_classes

    def text_to_embedding(self, raw_text):
        tokens = self.tokenizer(raw_text)
        vectorized = [self.vocab[token] for token in tokens if token in self.vocab.stoi]

        if len(vectorized) == 0:
            return torch.zeros(self.embedding_dim)

        # Average the embeddings
        vectorized = torch.stack(vectorized).mean(dim=0)
        return vectorized

    def one_hot_encode_labels(self, label):
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
        label_tensor[label - 1] = 1.0
        return label_tensor

    def text_records_to_embeddings(self, data_iter):
        feature_data = []
        labels = []
        for label, text in data_iter:
            feature_data.append(self.text_to_embedding(text))
            labels.append(self.one_hot_encode_labels(label))

        return feature_data, labels

    @staticmethod
    def save_data(data, filepath):
        torch.save(data, filepath)

    @staticmethod
    def load_data(filepath):
        if os.path.exists(filepath):
            return torch.load(filepath)
        return None

    def run(self, train_iter, test_iter, save_dir='vectorized_data') -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        train_data_path = os.path.join(save_dir, 'train_data.pt')
        test_data_path = os.path.join(save_dir, 'test_data.pt')

        train_data = DataPreprocessor.load_data(train_data_path)
        test_data = DataPreprocessor.load_data(test_data_path)

        if train_data is None or test_data is None:

            print("Vectorizing data...")

            train_feature_data, train_labels = self.text_records_to_embeddings(train_iter)
            test_feature_data, test_labels = self.text_records_to_embeddings(test_iter)

            DataPreprocessor.save_data((train_feature_data, train_labels), train_data_path)
            DataPreprocessor.save_data((test_feature_data, test_labels), test_data_path)
        else:

            print("Loading vectorized data from disk.")
            train_feature_data, train_labels = train_data
            test_feature_data, test_labels = test_data

        train_feature_data_tensor = torch.stack(train_feature_data)
        train_labels_tensor = torch.stack(train_labels)
        test_feature_data_tensor = torch.stack(test_feature_data)
        test_labels_tensor = torch.stack(test_labels)

        print("Data preprocessing complete.")
        return train_feature_data_tensor, train_labels_tensor, test_feature_data_tensor, test_labels_tensor

