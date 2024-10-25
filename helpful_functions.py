import torch

# Helper Function To Get Available Device and Assign It
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS available, being utilized")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA available, being utilized")
    elif torch.cpu.is_available():
        device = torch.device("cpu")
        print("CPU available, being utilized")
    return device
