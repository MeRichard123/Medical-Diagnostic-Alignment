import torch

def logtanh(x, eps=1e-8):
    # add small epsilon to avoid log(0) when x is near zero
    return torch.log(torch.tanh(x).abs() + eps)