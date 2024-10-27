import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import helpful_functions as hf


# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()

        # Encoder with an extra layer and dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Additional hidden layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),  # Latent space
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        # Decoder with an extra layer and dropout
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),  # Additional hidden layer
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return reconstructed, latent_space


def extract_features(train_loader, test_loader, input_size, hidden_size, device, learning_rate, num_epochs):
    # Creating Autoencoder
    autoencoder = Autoencoder(input_size, hidden_size).to(device)

    # Define loss function and optimizer with weight decay (L2 regularization)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)  # Apply L2 regularization

    # Learning rate scheduler (exponential decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    best_loss = np.inf
    early_stop_counter = 0
    all_losses = []

    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0

        for data in train_loader:
            inputs, _ = data  # Unpack inputs and ignore labels
            inputs = inputs.to(device)

            # Forward pass
            reconstructed, _ = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Learning rate scheduling step
        scheduler.step()

        # Average epoch loss
        epoch_loss = running_loss / len(train_loader)
        all_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop_counter = 0  # Reset counter if we get a new best loss
        else:
            early_stop_counter += 1  # Increment counter if no improvement

        # Stop training if patience is exceeded
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # Plot the training loss
    hf.autoencoder_plot_loss(all_losses)

    # Feature extraction on both train and test datasets
    train_features = []
    test_features = []

    autoencoder.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            latent_space = autoencoder.encoder(inputs)
            train_features.append(latent_space.cpu())

        for data in test_loader:
            inputs = data[0].to(device)
            latent_space = autoencoder.encoder(inputs)
            test_features.append(latent_space.cpu())

    train_features = torch.cat(train_features, dim=0)
    test_features = torch.cat(test_features, dim=0)

    return train_features, test_features