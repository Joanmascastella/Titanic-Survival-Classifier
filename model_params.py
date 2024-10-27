import torch
import torch.nn as nn

def define_svm_params(svm_model):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(svm_model.parameters(), lr=0.01)
    loss_list = []
    accuracy_list = []
    n_epochs = 100
    model = svm_model

    return model, optimizer, criterion, loss_list, accuracy_list, n_epochs

def define_knn_params(k_nearest_model):

    loss_list = []
    accuracy_list = []
    n_epochs = 100
    model = k_nearest_model

    return model, loss_list, accuracy_list, n_epochs