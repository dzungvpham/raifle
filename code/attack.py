import copy
import math
import random
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
        optimizer = optim.LBFGS([opt_params], line_search_fn="strong_wolfe", **kwargs)

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
            print("An exception occurred:", e)
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
            return (best_opt_params, best_loss)
        else:
            return (best_opt_params.sigmoid().round().long(), best_loss)
    else:
        if return_raw:
            return (best_opt_params[:num_items], best_opt_params[num_items:], best_loss)
        else:
            return (
                best_opt_params[:num_items].sigmoid().round().long(),
                best_opt_params[num_items:],
                best_loss,
            )

# Adaptation of IMIA code from https://github.com/hi-weiyuan/FedRec_IMIA/
def interaction_mia_fedrec(trainer, target_params, num_items, select_ratio=0.2):
    all_items = list(range(num_items))
    confirmed_pos = []
    confirmed_neg = []
    expected_pos_num = int(select_ratio * num_items)

    un_confirmed = copy.deepcopy(all_items)
    random.shuffle(un_confirmed)
    random_selected = un_confirmed[:expected_pos_num]

    while len(confirmed_pos) < expected_pos_num or len(un_confirmed) > expected_pos_num:
        new_ratings = torch.zeros(num_items)
        new_ratings[random_selected] = 1.0
        shadow_params = trainer(new_ratings)

        difference = F.pairwise_distance(shadow_params, target_params)
        indexes = torch.argsort(difference, descending=False).tolist()
        num = 0
        for item in indexes:
            if item not in confirmed_neg and item not in confirmed_pos:
                if item in random_selected:
                    confirmed_pos.append(item)
                else:
                    confirmed_neg.append(item)
                num += 1

            if num > (0.9 * num_items):
                break
        
        if len(confirmed_pos) >= expected_pos_num:
            break

        un_confirmed = list(set(all_items) - set(confirmed_neg + confirmed_pos))
        random.shuffle(un_confirmed)        
        random_selected = confirmed_pos + un_confirmed[:expected_pos_num - len(confirmed_pos)]
    
    final_ratings = torch.zeros(num_items)
    final_ratings[confirmed_pos] = 1.0
    return final_ratings