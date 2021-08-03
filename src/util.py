import torch

def to_device(data, device):
    """
    Transfer a tensor/list of tensor/tuple of tensor from cpu to a cuda machine.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, torch.device(device)) for x in data]
    return data.to(torch.device(device), non_blocking=True)