import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

def define_svm_params(svm_model):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(svm_model.parameters(), lr=0.005)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # Reduces LR every 20 epochs by half
    loss_list = []
    accuracy_list = []
    n_epochs = 100
    return svm_model, optimizer, criterion, scheduler, loss_list, accuracy_list, n_epochs

def define_knn_params(k_nearest_model):

    loss_list = []
    accuracy_list = []
    n_epochs = 100
    model = k_nearest_model

    return model, loss_list, accuracy_list, n_epochs