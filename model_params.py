import torch
import torch.nn as nn

def define_svm_params(svm_model):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(svm_model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    loss_list = []
    accuracy_list = []
    n_epochs = 200
    return svm_model, optimizer, criterion, scheduler, loss_list, accuracy_list, n_epochs

def define_knn_params(k_nearest_model):

    loss_list = []
    accuracy_list = []
    n_epochs = 100
    model = k_nearest_model

    return model, loss_list, accuracy_list, n_epochs