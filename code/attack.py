import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def reconstruct_interactions(
    trainer,
    target_params,
    num_items,
    lr=0.1,
    max_iters=100,
    return_raw=False,
):
    interaction_degree = nn.Parameter(torch.rand((1, num_items)) * 2 - 1)
    optimizer = optim.LBFGS([interaction_degree], lr=lr, max_iter=max_iters)

    def calc_loss():
        optimizer.zero_grad()
        interactions = interaction_degree.sigmoid()
        shadow_params = trainer(interactions)
        loss = F.mse_loss(shadow_params, target_params)
        loss.backward()
        return loss

    optimizer.step(calc_loss)

    if return_raw:
        return interaction_degree
    else:
        return interaction_degree.sigmoid().round().long().detach()
