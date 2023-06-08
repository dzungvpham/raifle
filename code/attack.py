import math
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
    num_rounds=10,
    return_raw=False,
    prior_penalty=None,
):
    if prior_penalty is None:
        prior_penalty = lambda _: torch.zeros(1)

    best_loss = math.inf
    global best_interaction_degree
    global best_private_params

    for _ in range(num_rounds):
        interaction_degree = nn.Parameter(torch.rand(num_items) * 2 - 1)
        optimizer = optim.LBFGS(
            [interaction_degree],
            lr=lr,
            max_iter=max_iters,
            line_search_fn="strong_wolfe",
        )

        def calc_loss():
            optimizer.zero_grad()
            interactions = interaction_degree.sigmoid()
            shadow_params = trainer(interactions)
            loss = F.mse_loss(shadow_params, target_params) + prior_penalty(
                interactions
            )
            loss.backward()
            return loss

        optimizer.step(calc_loss)

        cur_loss = optimizer.state[list(optimizer.state)[0]]["prev_loss"]

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_interaction_degree = interaction_degree

    if return_raw:
        return best_interaction_degree.detach()
    else:
        return best_interaction_degree.sigmoid().round().long().detach()


def reconstruct_interactions_with_private_params(
    trainer,
    target_params,
    num_items,
    private_params_size,
    lr=0.1,
    max_iters=100,
    num_rounds=10,
    return_raw=False,
    prior_penalty=None,
):
    if prior_penalty is None:
        prior_penalty = lambda _: torch.zeros(1)

    best_loss = math.inf
    global best_interaction_degree
    global best_private_params

    for _ in range(num_rounds):
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
            loss = F.mse_loss(shadow_params, target_params) + prior_penalty(
                interactions
            )
            loss.backward()
            return loss

        optimizer.step(calc_loss)

        cur_loss = optimizer.state[list(optimizer.state)[0]]["prev_loss"]

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_interaction_degree = interaction_degree
            best_private_params = private_params

    if return_raw:
        return (best_interaction_degree.detach(), best_private_params.detach())
    else:
        return (
            best_interaction_degree.sigmoid().round().long().detach(),
            best_private_params.detach(),
        )
