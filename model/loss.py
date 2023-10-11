import torch 

def L1_loss(output, target):
    return torch.nn.L1Loss()(output, target.view(-1, 1))
