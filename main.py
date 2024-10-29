# Import libraries
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Import other classes
import data as d
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
    train, test, y_train = d.clean_and_process_data(train, test)
    train_loader, val_loader = d.create_data_loaders(train, y_train)

    print("3. Compiling Models")
    # Compile SVM Classifier Model
    input_size = train.shape[1]
    svm_model = mc.SVMClassifier(input_size).to(device)
    s_mmodel, s_optimizer, s_criterion, s_scheduler, s_loss_list, s_accuracy_list, s_n_epochs = mp.define_svm_params(
        svm_model)

    # Compile K-Nearest Neighbours Model
    k_nearest_model = mc.KNNClassifier()
    knnmodel, k_loss_list, k_accuracy_list, k_n_epochs = mp.define_knn_params(k_nearest_model)

    print("4. Converting Train And Test Features To Tensors And Then To Data Loader Objects")

    # Convert train, test, and labels to PyTorch tensors
    train_x_tensor = torch.tensor(train.values, dtype=torch.float32).to(device)
    train_y_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device).unsqueeze(1)
    test_x_tensor = torch.tensor(test.values, dtype=torch.float32).to(device)

    # Create DataLoader objects
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    test_dataset = TensorDataset(test_x_tensor)  # Test dataset without labels
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("5. Training the Model")
    # Training SVM
    t.svm_train(s_mmodel, s_optimizer, s_criterion, s_scheduler, s_loss_list, s_accuracy_list, s_n_epochs,
                train_loader, val_loader, test_loader, submission)

    # Training KNN
    t.knn_train(knnmodel, k_loss_list, k_accuracy_list, k_n_epochs,
                                               train_loader, test_loader, submission)

    print("6. Training loops done. \n Predictions have been moved to the models respective submission files.")

if __name__ == '__main__':
    # Load Data
    train_file = "./data/train.csv"
    test_file = "./data/test.csv"
    submission_file = "./data/gender_submission.csv"
    device  = hf.get_device()
    main(train_file, test_file, submission_file, device)