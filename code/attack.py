import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def reconstruct_interactions(
    trainer,
    target_params,
    num_items,
    private_params_size=0,
    lr=0.1,
    max_iters=100,
    num_rounds=10,
    return_raw=False,
    prior_penalty=None,
):
    if prior_penalty is None:
        prior_penalty = lambda _: torch.zeros(1)

    best_loss = math.inf
    global best_opt_params

    for _ in range(num_rounds):
        opt_params = nn.Parameter(torch.rand(num_items + private_params_size) * 2 - 1)
        optimizer = optim.LBFGS(
            [opt_params],
            lr=lr,
            max_iter=max_iters,
            line_search_fn="strong_wolfe",
        )

        def calc_loss():
            optimizer.zero_grad()
            interactions = opt_params[:num_items].sigmoid()
            shadow_params = (
                trainer(interactions)
                if private_params_size == 0
                else trainer(interactions, opt_params[num_items:])
            )
            loss = F.mse_loss(shadow_params, target_params) + prior_penalty(
                interactions
            )
            loss.backward()
            return loss

        optimizer.step(calc_loss)

        cur_loss = optimizer.state[list(optimizer.state)[0]]["prev_loss"]

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_opt_params = opt_params

    if private_params_size == 0:
        if return_raw:
            return best_opt_params.detach()
        else:
            return best_opt_params.sigmoid().round().long().detach()
    else:
        if return_raw:
            return (best_opt_params[:num_items].detach(), best_opt_params[num_items:].detach())
        else:
            return (
                best_opt_params[:num_items].sigmoid().round().long().detach(),
                best_opt_params[num_items:].detach(),
            )

# Reproduction of the interaction inference attack method in https://arxiv.org/pdf/2301.10964.pdf
def interaction_mia_fedrec(
    trainer,
    target_params,
    num_items,
    pos_ratio=0.25,
    select_ratio=0.2,
):
    best_guess = torch.zeros(num_items)

    while best_guess.sum() < pos_ratio * num_items:
        guess = best_guess.logical_or(torch.bernoulli(torch.ones(num_items) * pos_ratio))
        shadow_params = trainer(guess.long())
        dist = (shadow_params - target_params).pow(2).sum(dim=1).sqrt()
        selected_guess = guess.logical_and(dist <= dist.quantile(select_ratio))
        best_guess = best_guess.logical_or(selected_guess)

    return best_guess
