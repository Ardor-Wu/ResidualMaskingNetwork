import torch


def visualize(model):
    loss_fn = torch.nn.MSELoss(reduction='sum')

