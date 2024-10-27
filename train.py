import torch

import helpful_functions as hf

device = hf.get_device()


def svm_train(model, optimizer, criterion, loss_list,
              accuracy_list, n_epochs, train_loader,
              test_loader, submission):
    # Training Loop
    def train(n_epochs):
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            accurate_predictions = 0.0
            total_predictions = 0.0
            accuracy_threshold = 0.1

            for x, y in train_loader:  # Use train_loader as a DataLoader
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                y_pred = model(x)
                loss = criterion(y_pred, y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss and calculate accuracy
                running_loss += loss.item()
                accurate_predictions += torch.sum(torch.abs(y_pred - y) <= (accuracy_threshold * y)).item()
                total_predictions += y.size(0)

            # Average loss
            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = (accurate_predictions / total_predictions) * 100  # Convert to percentage

            # Append to lists for tracking
            loss_list.append(epoch_loss)
            accuracy_list.append(epoch_accuracy)

            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    train(n_epochs)

    return loss_list, accuracy_list


def knn_train(model, loss_list, accuracy_list, n_epochs,
              train_features, test_features, submission):
    return