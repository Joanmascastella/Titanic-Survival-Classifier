# Import libraries
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Import other classes
import data as d
import feature_extractor as ft
import helpful_functions as hf
import model_complier as mc
import model_params as mp
import train as t


def main(train_file, test_file, submission_file, device):

    # Save File Contents
    print("1. Loading & Saving Data To Data Frames")
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    submission = pd.read_csv(submission_file)
    device = device


    # Clean & Process Data
    print("2. Cleaning & Processing Data")
    train_cleaned, test_cleaned, y_train = d.clean_data(train, test)
    train_processed, test_processed = d.process_data(train_cleaned, test_cleaned)
    train_loader, test_loader = d.create_data_loaders(train_processed, test_processed, y_train)

    # Extract Features
    print("3. Extracting Features Using Auto Encoder")
    input_size = train_processed.shape[1]  # Number of input features
    hidden_size = 40
    learning_rate = 0.001
    num_epochs = 120

    # Extract features using the autoencoder
    train_features, test_features = ft.extract_features(train_loader, test_loader,
                                                        input_size, hidden_size, device,
                                                        learning_rate, num_epochs)

    print("4. Compiling Models")
    # Compile SVM Classifier Model
    input_size = train_features.shape[1]
    svm_model = mc.SVMClassifier(input_size).to(device)
    s_mmodel, s_optimizer, s_criterion, s_loss_list, s_accuracy_list, s_n_epochs = mp.define_svm_params(svm_model)

    # Compile K-Nearest Neighbours Model
    k_nearest_model = mc.KNNClassifier()
    knnmodel, k_loss_list, k_accuracy_list, k_n_epochs = mp.define_knn_params(k_nearest_model)

    print("5. Converting Train And Test Features To Tensors And Then To Data Loader Objects")
    # Converting To Tensors
    train_x_tensor = train_features.clone().detach()  # Use clone().detach() to avoid warning
    train_y_tensor = torch.tensor(y_train, dtype=torch.float).to(device).unsqueeze(1)
    test_x_tensor = test_features.clone().detach()

    # Creating DataLoader Objects
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)  # Create TensorDataset for train
    test_dataset = TensorDataset(test_x_tensor)  # Create TensorDataset for test

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("6. Training the Model")
    # Training SVM
    s_accuracy_list, s_loss_list = t.svm_train(s_mmodel, s_optimizer, s_criterion,
                                               s_loss_list, s_accuracy_list, s_n_epochs,
                                               train_loader, test_loader, submission)

    # Training KNN
    k_accuracy_list, k_loss_list = t.knn_train(knnmodel, k_loss_list, k_accuracy_list, k_n_epochs,
                                               train_loader, test_loader, submission)

    print("7. Comparing Results")
    print(f" SVM Accuracy: {s_accuracy_list[-1] * 100:.2f}% \n")
    # add plot
    print(f" SVM Loss: {s_loss_list[-1] * 100:.2f}% \n")

    print(f" KNN Accuracy: {k_accuracy_list[-1] * 100:.2f}% \n")
    # add plot
    print(f" KNN Loss: {k_loss_list[-1] * 100:.2f}% \n")


if __name__ == '__main__':
    # Load Data
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    submission_file = "./data/gender_submission.csv"
    device  = hf.get_device()
    main(train_file, test_file, submission_file, device)