# Import libraries
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Import other classes
import data as d
import feature_extractor as ft
import model_complier as mc
import helpful_functions as hf
import model_params as mp

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
    train_features, test_features = ft.extract_features(train_loader, test_loader, input_size, hidden_size, device, learning_rate, num_epochs)


    print("4. Compiling Models")
    # Compile SVM Classifier Model
    input_size = train_features.shape[1]
    svm_model = mc.SVMClassifier(input_size)
    model, optimizer, criterion, loss_list, accuracy_list, n_epochs = mp.define_svm_params(svm_model)

    # Compile K-Nearest Neighbours Model
    k_nearest_model = mc.KNNClassifier()
    model, loss_list, accuracy_list, n_epochs = mp.define_knn_params(k_nearest_model)

    print("5. Training the Model")




    print("6. Comparing Results")



if __name__ == '__main__':
    # Load Data
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    submission_file = "./data/gender_submission.csv"
    device  = hf.get_device()
    main(train_file, test_file, submission_file, device)