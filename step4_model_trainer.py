import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pipeline_node_abstract import PipelineNode
from util.plot_util import plot_stat_curves


class TextClassifier(nn.Module):
    def __init__(self, embedding_dim=100, num_classes=4):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 200)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class ModelTrainer(PipelineNode):
    def __init__(self, model=TextClassifier(), learning_rate=0.001, batch_size=128, num_epochs=20):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # Record stats
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_precisions = []
        self.test_precisions = []
        self.train_recalls = []
        self.test_recalls = []
        self.train_f1s = []
        self.test_f1s = []

    def run(self, train_data, train_labels, test_data, test_labels):

        train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=self.batch_size, shuffle=False)

        print("Initial Stats: ")
        self.print_stats(train_loader, test_loader)

        for epoch in range(self.num_epochs):
            self.model.train()
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                loss = self.calculate_batch_loss(inputs, labels)
                loss.backward()
                self.optimizer.step()

            train_loss = self.calculate_overall_loss(train_loader)
            test_loss = self.calculate_overall_loss(test_loader)

            train_accuracy, train_precision, train_recall, train_f1 = self.evaluate(train_loader)
            test_accuracy, test_precision, test_recall, test_f1 = self.evaluate(test_loader)

            self.record_stats(train_loss, test_loss, train_accuracy, train_precision, train_recall, train_f1,
                              test_accuracy, test_precision, test_recall, test_f1)

            print(
                f'Epoch {epoch + 1}/{self.num_epochs}, \n'
                f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                f'Train Precision: {train_precision:.4f}, '
                f'Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, \n'                    
                f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, '
                f'Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}'
            )

        print("Final Stats: ")
        self.print_stats(train_loader, test_loader)
        self.plot_recorded_stats()

        # The model is ready for next steps - save it
        torch.save(self.model.state_dict(),
                   os.path.join("model_repo", f"model_state_dict_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pth"))


    def evaluate(self, data_loader):
        self.model.eval()
        all_predictions = []
        all_true_classes = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                _, true_classes = torch.max(labels, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_true_classes.extend(true_classes.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_true_classes = np.array(all_true_classes)

        accuracy = accuracy_score(all_true_classes, all_predictions)
        precision = precision_score(all_true_classes, all_predictions, average='macro')
        recall = recall_score(all_true_classes, all_predictions, average='macro')
        f1 = f1_score(all_true_classes, all_predictions, average='macro')

        return accuracy, precision, recall, f1

    def record_stats(self, train_loss, test_loss, train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision,
                     test_recall, test_f1):

        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)
        self.train_precisions.append(train_precision)
        self.test_precisions.append(test_precision)
        self.train_recalls.append(train_recall)
        self.test_recalls.append(test_recall)
        self.train_f1s.append(train_f1)
        self.test_f1s.append(test_f1)

    def plot_recorded_stats(self):

        plot_stat_curves(self.train_losses, self.test_losses, "Loss")
        plot_stat_curves(self.train_accuracies, self.test_accuracies, "Accuracy")
        plot_stat_curves(self.train_precisions, self.test_precisions, "Precision")
        plot_stat_curves(self.train_recalls, self.test_recalls, "Recall")
        plot_stat_curves(self.train_f1s, self.test_f1s, "F1")


    def calculate_batch_loss(self, inputs, labels):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return loss

    def calculate_overall_loss(self, data_loader):
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                loss = self.calculate_batch_loss(inputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        return total_loss / total_samples

    def print_stats(self, train_loader, test_loader):

        train_acc, train_prec, train_recall, train_f1 = self.evaluate(train_loader)
        test_acc, test_prec, test_recall, test_f1  = self.evaluate(test_loader)
        print(
            f'Train Acc: {train_acc:.4f}, rain Precision: {train_prec:.4f}, '
            f'Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}')

        print(
            f'Test Acc: {test_acc:.4f}, Test Precision: {test_prec:.4f}, '
            f'Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')




