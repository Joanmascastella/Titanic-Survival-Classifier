import torch
import torch.nn as nn

# SVM Classifier
class SVMClassifier(nn.Module):
    def __init__(self, input_size):
        super(SVMClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)  # Two outputs for CrossEntropyLoss

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.fc2(x)  # Two output logits for each class
        return x

# K-Nearest Neighbours Classifier
class KNNClassifier:
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.metric = metric  # Allow different distance metrics

    def fit(self, X_train, y_train):
        # Store training data
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Calculate distance based on chosen metric
        if self.metric == "euclidean":
            distances = torch.cdist(X_test, self.X_train)
        elif self.metric == "manhattan":
            distances = torch.cdist(X_test, self.X_train, p=1)
        else:
            raise ValueError("Unsupported metric. Choose 'euclidean' or 'manhattan'.")

        # Get the k closest training samples for each test sample
        _, indices = torch.topk(distances, self.k, largest=False)

        # Gather labels of these k closest samples
        nearest_labels = self.y_train[indices]

        # Perform mode calculation on the CPU
        predictions = torch.mode(nearest_labels.cpu(), dim=1).values.to(X_test.device)  # Move back to MPS if needed
        return predictions