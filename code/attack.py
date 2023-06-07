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
    prior_penalty=None,
):
    if prior_penalty is None:
        prior_penalty = lambda _: torch.zeros(1)

    interaction_degree = nn.Parameter(torch.rand(num_items) * 2 - 1)
    optimizer = optim.LBFGS(
        [interaction_degree], lr=lr, max_iter=max_iters, line_search_fn="strong_wolfe"
    )

    def calc_loss():
        optimizer.zero_grad()
        interactions = interaction_degree.sigmoid()
        shadow_params = trainer(interactions)
        loss = F.mse_loss(shadow_params, target_params) + prior_penalty(interactions)
        loss.backward()
        return loss

    optimizer.step(calc_loss)

    if return_raw:
        return interaction_degree
    else:
        return interaction_degree.sigmoid().round().long().detach()


def reconstruct_interactions_with_private_params(
    trainer,
    target_params,
    num_items,
    private_params_size,
    lr=0.1,
    max_iters=100,
    return_raw=False,
    prior_penalty=None,
):
    if prior_penalty is None:
        prior_penalty = lambda _: torch.zeros(1)
    
    interaction_degree = nn.Parameter(torch.rand(num_items) * 2 - 1)
    private_params = nn.Parameter(torch.rand(private_params_size) * 2 - 1)
    optimizer = optim.LBFGS(
        [interaction_degree, private_params],
        lr=lr,
        max_iter=max_iters,
        line_search_fn="strong_wolfe",
    )

    def calc_loss():
        optimizer.zero_grad()
        interactions = interaction_degree.sigmoid()
        shadow_params = trainer(interactions, private_params)
        loss = F.mse_loss(shadow_params, target_params) + prior_penalty(interactions)
        loss.backward()
        return loss

    optimizer.step(calc_loss)

    if return_raw:
        return (interaction_degree.detach(), private_params.detach())
    else:
        return (
            interaction_degree.sigmoid().round().long().detach(),
            private_params.detach(),
        )
