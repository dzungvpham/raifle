import copy
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchopt
import traceback
from cmaes import CMA
from tqdm.notebook import tqdm

def reconstruct_interactions(
    trainer,
    target_params,
    num_items,
    private_params_size=0,
    loss_fn=F.mse_loss,
    num_rounds=1,
    return_raw=False,
    prior_penalty=None,
    device=torch.device("cpu"),
    **kwargs,
):
    if prior_penalty is None:
        prior_penalty = lambda _: 0.0

    best_loss = math.inf
    best_opt_params = None

    for _ in range(num_rounds):
        opt_params = nn.Parameter(torch.rand(num_items + private_params_size, device=device) * 2 - 1)
        optimizer = optim.LBFGS([opt_params], line_search_fn="strong_wolfe", **kwargs)

        def calc_loss():
            optimizer.zero_grad()
            interactions = opt_params[:num_items].sigmoid()
            shadow_params = (
                trainer(interactions)
                if private_params_size == 0
                else trainer(interactions, opt_params[num_items:])
            )
            loss = loss_fn(shadow_params, target_params) + prior_penalty(interactions)
            loss.backward(inputs=[opt_params])
            return loss

        try:
            optimizer.step(calc_loss)
        except Exception:
            print("An exception occurred in the optimization step!")
            traceback.print_exc()
            continue

        optimizer_state = optimizer.state[list(optimizer.state)[0]]
        if "prev_loss" not in optimizer_state:
            print("Optimization did not take any step!")
            continue
        else:
            cur_loss = optimizer_state["prev_loss"]

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_opt_params = opt_params.detach()

    if best_opt_params is None:
        print("Optimization failed! Defaulting to random guessing.")
        best_opt_params = torch.rand(num_items + private_params_size) * 2 - 1

    if private_params_size == 0:
        if return_raw:
            return best_opt_params, best_loss
        else:
            return best_opt_params.sigmoid().round().long(), best_loss
    else:
        if return_raw:
            return best_opt_params[:num_items], best_opt_params[num_items:], best_loss
        else:
            return (
                best_opt_params[:num_items].sigmoid().round().long(),
                best_opt_params[num_items:],
                best_loss,
            )

def reconstruct_interactions_adam(
    trainer,
    target_params,
    num_items,
    loss_fn=F.mse_loss,
    num_epochs=1,
    device=torch.device("cpu"),
    progress_bar=False,
    **kwargs,
):
    opt_params = nn.Parameter(torch.rand(num_items, device=device) * 2 - 1)
    optimizer = optim.Adam([opt_params], **kwargs)
    best_loss = math.inf
    best_params = None
    iterator = tqdm(range(num_epochs+1)) if progress_bar else range(num_epochs+1)
    for i in iterator:
        optimizer.zero_grad()
        shadow_params = trainer(opt_params.sigmoid())
        loss = loss_fn(shadow_params, target_params)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = opt_params.detach()
        if progress_bar:
            iterator.set_description(f"Loss: {round(loss.detach().item(), 5)}")
        if i == num_epochs:
            break
        loss.backward(inputs=[opt_params])
        optimizer.step()
    return best_params


"""
    Functional reconstruction without in-place modification for optimized ADM
"""
def reconstruct_interactions_functional(
    trainer,
    target_params,
    num_items,
    loss_fn=F.mse_loss,
    num_epochs=1,
    inplace=False,
    device=torch.device("cpu"),
    **kwargs,
):
    optimizer = torchopt.FuncOptimizer(torchopt.adam(**kwargs), inplace=inplace)
    opt_params = (nn.Parameter(torch.rand(num_items, device=device) * 2 - 1),)
    for _ in range(num_epochs):
        shadow_params = trainer(opt_params[0].sigmoid())
        loss = loss_fn(shadow_params, target_params)
        opt_params = optimizer.step(loss, opt_params)
    return opt_params[0]


def optimize_ltr_data_manipulation_es(
    trainer,
    model,
    train_dataset,
    num_query,
    num_features,
    max_num_items,
    max_epochs=1,
    num_reconstructions_per_step=1,
    bounds=np.array([[0.0, 0.0], [0.0, 1.0]]),
    seed=None,
    **kwargs
):
    optimizer = CMA(mean=np.zeros(2), sigma=1.0, bounds=bounds, seed=seed)
    masks = [torch.ones((max_num_items, num_features)) for _ in range(num_query)]
    num_features_keep = num_features // num_query
    for idx, mask in enumerate(masks):
        mask[:,:(idx * num_features_keep)].zero_()
        mask[:,((idx + 1) * num_features_keep):].zero_()
    
    query_ids = train_dataset.get_all_query_ids()

    def batch_trainer(model_params, grouped_train_data, interactions, indices):
        target_params = []
        num_items = interactions.shape[0] // num_reconstructions_per_step
        for i in range(num_reconstructions_per_step):
            I = interactions[(i * num_items):((i + 1) * num_items)]
            new_grouped_data = [(F, R, I[indices[idx][0]:indices[idx][1]]) for idx, (F, R) in enumerate(grouped_train_data)]
            params = trainer(model, model_params, new_grouped_data)
            target_params.append(params)
        return torch.cat([p.view(-1) for p in target_params])
    
    best_params = None
    best_loss = math.inf

    for epoch in (pbar := tqdm(range(max_epochs))):
        solns = []
        for _ in range(optimizer.population_size):
            perturb_params = optimizer.ask()
            params = model.gen_params()
            grouped_train_data = []
            indices = []
            num_items_total = 0
            while len(grouped_train_data) < num_query:
                grouped_train_data = []
                indices = []
                num_items_total = 0  
                qids = random.sample(query_ids, num_query)
                grouped_data = train_dataset.get_data_for_queries(list(qids))

                for idx, (relevances, features) in enumerate(grouped_data):
                    if len(relevances) == 1:
                        break
                    features = torch.Tensor(features)
                    ranking = model.rank(params, features, sample=True)[:max_num_items]
                    features = features[ranking]
                    # Remap the original ranking into the correct range
                    _, ranking = torch.where(
                        torch.sort(ranking)[0].unsqueeze(1) == ranking.unsqueeze(0)
                    )
                    num_items = len(ranking)
                    mask = masks[idx][:num_items,:]
                    features = features * mask + torch.normal(perturb_params[0], perturb_params[1], mask.shape) * (1.0 - mask)
                    grouped_train_data.append((features, ranking))
                    indices.append((num_items_total, num_items_total + num_items))
                    num_items_total += num_items
            
            target_interactions = torch.randint(0, 2, (num_items_total * num_reconstructions_per_step,)).float()
            target_params = batch_trainer(params, grouped_train_data, target_interactions, indices)

            guessed_interactions, _ = reconstruct_interactions(
                lambda I: batch_trainer(params, grouped_train_data, I, indices),
                target_params,
                num_items_total * num_reconstructions_per_step,
                return_raw=True,
                **kwargs,
            )

            loss = F.binary_cross_entropy_with_logits(guessed_interactions, target_interactions)
            solns.append((perturb_params, loss.item()))

        avg_loss = np.mean([s[1] for s in solns])
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = optimizer._mean.copy()
                           
        optimizer.tell(solns)
        pbar.set_description(f"Epoch {epoch}: Params: {round(optimizer._mean[0], 3)}, {round(optimizer._mean[1], 3)} | Loss: {round(avg_loss, 3)}")
        if optimizer.should_stop():
            break

    return best_params, masks


def optimize_ltr_data_manipulation_grad(
    trainer,
    model,
    train_dataset,
    num_query,
    num_features,
    max_num_items,
    epochs=1,
    loss_fn=F.binary_cross_entropy_with_logits,
    l1_factor=0.0,
    max_grad_l1_norm=None,
    grad_clip=None,
    num_reconstructions_per_step=1,
    reconstruct_epochs=1,
    reconstruct_lr=0.1,
    **kwargs
):
    masks = [torch.ones((max_num_items, num_features)) for _ in range(num_query)]
    num_features_keep = num_features // num_query
    for idx, mask in enumerate(masks):
        mask[:,:(idx * num_features_keep)].zero_()
        mask[:,((idx + 1) * num_features_keep):].zero_()

    perturb = nn.Parameter(torch.normal(0, 0.1, masks[0].shape))
    query_ids = train_dataset.get_all_query_ids()
    optimizer = optim.Adam([perturb], **kwargs)

    def batch_trainer(model_params, grouped_train_data, interactions, indices):
        target_params = []
        num_items = interactions.shape[0] // num_reconstructions_per_step
        for i in range(num_reconstructions_per_step):
            I = interactions[(i * num_items):((i + 1) * num_items)]
            new_grouped_data = [(F, R, I[indices[idx][0]:indices[idx][1]]) for idx, (F, R) in enumerate(grouped_train_data)]
            params = trainer(model, model_params, new_grouped_data)
            target_params.append(params)
        return torch.cat([p.view(-1) for p in target_params])

    for _ in (pbar := tqdm(range(epochs))):
        optimizer.zero_grad()
        params = model.gen_params()
        grouped_train_data = []
        indices = []
        num_items_total = 0
        while len(grouped_train_data) < num_query:
            grouped_train_data = []
            indices = []
            num_items_total = 0  
            qids = random.sample(query_ids, num_query)
            grouped_data = train_dataset.get_data_for_queries(list(qids))

            for idx, (relevances, features) in enumerate(grouped_data):
                if len(relevances) == 1:
                    break
                features = torch.Tensor(features)
                ranking = model.rank(params, features, sample=True)[:max_num_items]
                features = features[ranking]
                # Remap the original ranking into the correct range
                _, ranking = torch.where(
                    torch.sort(ranking)[0].unsqueeze(1) == ranking.unsqueeze(0)
                )
                num_items = len(ranking)
                mask = masks[idx][:num_items,:]
                features = features * mask + perturb[:num_items,:] * (1.0 - mask)
                grouped_train_data.append((features, ranking))
                indices.append((num_items_total, num_items_total + num_items))
                num_items_total += num_items
        
        target_interactions = torch.randint(0, 2, (num_items_total * num_reconstructions_per_step,)).float()
        target_params = batch_trainer(params, grouped_train_data, target_interactions, indices)

        guessed_interactions = reconstruct_interactions_functional(
            lambda I: batch_trainer(params, grouped_train_data, I, indices),
            target_params,
            num_items_total * num_reconstructions_per_step,
            num_epochs=reconstruct_epochs,
            lr=reconstruct_lr,
            eps_root=1e-08, # Must be set
            use_accelerated_op=False, # True might not work
        )

        loss = loss_fn(guessed_interactions, target_interactions) + l1_factor * torch.linalg.vector_norm(perturb, ord=1)
        loss.backward(inputs=[perturb])
        if max_grad_l1_norm is not None:
            nn.utils.clip_grad_norm_(perturb, max_grad_l1_norm, norm_type=1.0, error_if_nonfinite=True)
        if grad_clip is not None:
            nn.utils.clip_grad_value_(perturb, grad_clip)
        optimizer.step()

        pbar.set_description(f"Loss: {round(loss.detach().item(), 3)}")

    return perturb.detach(), masks


def optimize_image_manipulation(
    data, target, feature_extractor,
    max_epochs=100,
    loss_fn=F.mse_loss,
    early_stop=1e-03,
    linf_factor=0.0,
    progress_bar=False,
    **kwargs,
):
    opt_data = torch.clone(data).requires_grad_(True)
    optimizer = optim.Adam([opt_data], **kwargs)
    max_vals = [(1.0 - 0.485)/0.229, (1.0 - 0.456)/0.224, (1.0 - 0.406)/0.225]
    min_vals = [-0.485/0.229, -0.456/0.224, -0.406/0.225]

    iterator = tqdm(range(max_epochs)) if progress_bar else range(max_epochs)
    for _ in iterator:
        optimizer.zero_grad()
        extracted_features = feature_extractor(opt_data)
        linf_loss = linf_factor * torch.linalg.vector_norm(opt_data - data, ord=float('inf'))
        loss = loss_fn(extracted_features, target) + linf_loss
        loss.backward(inputs=[opt_data])
        optimizer.step()
        with torch.no_grad():
            for c in range(3):
                opt_data[:, c].clip_(min = min_vals[c], max=max_vals[c])
        if progress_bar:
            iterator.set_description(f"Loss: {round(loss.detach().item(), 5)}")

        if early_stop is not None and loss.item() < early_stop:
            break

    return opt_data.detach()


def optimize_image_manipulation_batches(
    data, target, feature_extractor, batch_size=128, **kwargs,
):
    num_data = data.shape[0]
    num_batches = int(math.ceil(num_data / batch_size))
    batch_res = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_data = data[start:end,:]
        batch_target = target[start:end,:]
        batch_opt = optimize_image_manipulation(batch_data, batch_target, feature_extractor, **kwargs)
        batch_res.append(batch_opt)
    return torch.vstack(batch_res)


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