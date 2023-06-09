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
