import torch
import torch.nn as nn

# SVM Classifier
class SVMClassifier(nn.Module):
    def __init__(self, input_size):
        super(SVMClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Hidden layer
        self.fc2 = nn.Linear(16, 1)  # Single output for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits
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

        # Determine the most common label among the nearest neighbors
        predictions = torch.mode(nearest_labels, dim=1).values
        return predictions