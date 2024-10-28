import torch

import helpful_functions as hf

device = hf.get_device()


def svm_train(model, optimizer, criterion, scheduler, loss_list,
              accuracy_list, n_epochs, train_loader,
              test_loader, submission):
    def train(n_epochs):
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # Forward pass
                y_pred = model(x)
                loss = criterion(y_pred, y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate

                running_loss += loss.item()

                # Apply a threshold to logits for binary prediction (e.g., 0.5 if using BCEWithLogitsLoss)
                predicted_labels = (torch.sigmoid(y_pred) > 0.5).float()
                correct_predictions += (predicted_labels == y).sum().item()
                total_predictions += y.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_predictions * 100  # Percentage

            loss_list.append(epoch_loss)
            accuracy_list.append(epoch_accuracy)

            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    train(n_epochs)
    return loss_list, accuracy_list


def knn_train(model, loss_list, accuracy_list, n_epochs, train_loader, test_loader, submission):
    for epoch in range(n_epochs):
        correct_predictions = 0
        total_predictions = 0
        running_loss = 0

        for train_batch, labels in train_loader:
            model.fit(train_batch, labels)  # Fit model to training batch
            predictions = model.predict(train_batch)  # Predict on training batch

            loss = torch.mean((predictions - labels) ** 2).item()  # Compute MSE loss
            running_loss += loss

            correct_predictions += torch.sum(predictions == labels).item()  # Count correct predictions
            total_predictions += labels.size(0)

        epoch_accuracy = (correct_predictions / total_predictions) * 100
        epoch_loss = running_loss / len(train_loader)

        accuracy_list.append(epoch_accuracy)
        loss_list.append(epoch_loss)

        print(f"Epoch [{epoch + 1}/{n_epochs}], KNN Loss: {epoch_loss:.4f}, KNN Accuracy: {epoch_accuracy:.2f}%")

    return loss_list, accuracy_list
