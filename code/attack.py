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
    loss_func=F.mse_loss,
    num_rounds=1,
    return_raw=False,
    prior_penalty=None,
    **kwargs,
):
    if prior_penalty is None:
        prior_penalty = lambda _: torch.zeros(1)

    best_loss = math.inf
    global best_opt_params

    for _ in range(num_rounds):
        opt_params = nn.Parameter(torch.rand(num_items + private_params_size) * 2 - 1)
        optimizer = optim.LBFGS(
            [opt_params],
            line_search_fn="strong_wolfe",
            **kwargs,
        )

        def calc_loss():
            optimizer.zero_grad()
            interactions = opt_params[:num_items].sigmoid()
            shadow_params = (
                trainer(interactions)
                if private_params_size == 0
                else trainer(interactions, opt_params[num_items:])
            )
            loss = loss_func(shadow_params, target_params) + prior_penalty(interactions)
            loss.backward()
            return loss

        try:
            optimizer.step(calc_loss)
        except Exception as e:
            print(e)
            continue

        optimizer_state = optimizer.state[list(optimizer.state)[0]]
        if "prev_loss" not in optimizer_state:
            continue
        else:
            cur_loss = optimizer_state["prev_loss"]

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_opt_params = opt_params.detach()

    if private_params_size == 0:
        if return_raw:
            return best_opt_params
        else:
            return best_opt_params.sigmoid().round().long()
    else:
        if return_raw:
            return (best_opt_params[:num_items], best_opt_params[num_items:])
        else:
            return (
                best_opt_params[:num_items].sigmoid().round().long(),
                best_opt_params[num_items:],
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
