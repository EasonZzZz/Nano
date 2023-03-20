import torch

USE_CUDA = torch.cuda.is_available()


def FloatTensor(x, device=0):
    if USE_CUDA:
        return torch.cuda.FloatTensor(x).to(device)
    else:
        return torch.FloatTensor(x)


def LongTensor(x, device=0):
    if USE_CUDA:
        return torch.cuda.LongTensor(x).to(device)
    else:
        return torch.LongTensor(x)
