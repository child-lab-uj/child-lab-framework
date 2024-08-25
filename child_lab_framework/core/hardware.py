import torch

def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')

    if torch.mps.device_count() > 0:
        return torch.device('mps')

    return torch.device('cpu')
