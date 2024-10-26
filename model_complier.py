import torch
import torch.nn as nn

# SVM Classifier
class SVMClassifier(nn.Module):
    def __init__(self, input_size):
        super(SVMClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # First hidden layer with 16 units
        self.fc2 = nn.Linear(16, 10)  # Second hidden layer with 10 units
        self.fc3 = nn.Linear(10, 2)   # Output layer with 2 units for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

# K-Nearest Neighbours Classifier
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        # Store training data
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        # Calculate Euclidean distance between test samples and all training samples
        distances = torch.cdist(X_test, self.X_train)  # Shape: [n_test_samples, n_train_samples]

        # Get the k closest training samples for each test sample
        _, indices = torch.topk(distances, self.k, largest=False)

        # Gather labels of these k closest samples
        nearest_labels = self.y_train[indices]  # Shape: [n_test_samples, k]

        # Determine the most common label among the nearest neighbors
        predictions = torch.mode(nearest_labels, dim=1).values  # Shape: [n_test_samples]

        return predictions