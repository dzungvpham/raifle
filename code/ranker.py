import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap


class BaseRanker:
    def __init__(self):
        pass

    def gen_params(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def grad(self):
        raise NotImplementedError


class BasePDGDRanker(BaseRanker):
    def forward(self, params, features):
        raise NotImplementedError

    def forward_multiple(self, params, features):
        return vmap(self.forward, in_dims=(None, 0))(*(params, features))

    def log_ranking_prob(self, ranking, fx):
        return -torch.logcumsumexp(fx[ranking.flip(dims=(0,))], dim=0).sum()

    def calc_log_pos_bias_weight(self, ranking, fx, num_items):
        def flip_ranking(item_pair):
            item1, item2 = item_pair
            item1_idx, item2_idx = (ranking == item1).long().argmax(), (
                ranking == item2
            ).long().argmax()
            flipped_ranking = ranking.clone()
            flipped_ranking[item1_idx], flipped_ranking[item2_idx] = item2, item1
            return flipped_ranking

        log_prob_swapped = torch.tensor(
            list(
                map(
                    lambda pair: self.log_ranking_prob(flip_ranking(pair), fx),
                    itertools.combinations(range(num_items), 2),
                )
            )
        )
        log_prob_swapped_mtx = torch.zeros(num_items, num_items)
        log_prob_swapped_mtx[
            torch.triu_indices(num_items, num_items, offset=1).tolist()
        ] = log_prob_swapped
        log_prob_swapped_mtx = log_prob_swapped_mtx + log_prob_swapped_mtx.t()

        log_prob_orig = self.log_ranking_prob(ranking, fx)
        return log_prob_swapped_mtx - torch.logaddexp(
            log_prob_orig, log_prob_swapped_mtx
        )

    def grad(self, params, features, ranking, interactions, log_pos_bias_weight=None):
        num_items = features.shape[0]
        fx = self.forward_multiple(params, features)

        # Calculate position bias reweighing
        if log_pos_bias_weight is None:
            log_pos_bias_weight = self.calc_log_pos_bias_weight(ranking, fx, num_items)

        # Calculate gradient of probabilities of click pairs
        fx_expanded = fx.expand(num_items, -1)
        fx_sum = fx_expanded.t() + fx_expanded
        fx_logsumexp = torch.logsumexp(torch.cartesian_prod(fx, fx), dim=1).reshape(
            num_items, num_items
        )
        fx_grad = vmap(grad(self.forward), in_dims=(None, 0))(*(params, features))
        fx_grad_diff = vmap(torch.sub, in_dims=(0, None))(*(fx_grad, fx_grad))

        # Calculate final gradients
        interactions = interactions.reshape(1, -1)
        interaction_matrix = interactions.t() @ (1 - interactions)
        weights = (log_pos_bias_weight + fx_sum - 2 * fx_logsumexp).exp()
        res = interaction_matrix.unsqueeze(2) * weights.unsqueeze(2) * fx_grad_diff

        return res.sum(dim=(0, 1))


class LinearPDGDRanker(BasePDGDRanker):
    def __init__(self, feature_size):
        self.feature_size = feature_size

    def gen_params(self):
        return torch.rand(self.feature_size) * 2 - 1

    def forward(self, params, features):
        return params.dot(features)


class Neural1LayerPDGDRanker(BasePDGDRanker):
    def __init__(self, feature_size, hidden_size, activation=F.relu):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.activation = activation

    def gen_params(self):
        return torch.rand((self.feature_size + 1) * self.hidden_size) * 2 - 1

    def forward(self, params, features):
        num_hidden_features = self.feature_size * self.hidden_size
        hidden_params = params[:num_hidden_features].reshape(self.hidden_size, -1).t()
        res = self.activation(features @ hidden_params)
        return params[num_hidden_features:].dot(res)


class Neural2LayerPDGDRanker(BasePDGDRanker):
    def __init__(self, feature_size, hidden_size, hidden_size2, activation=F.relu):
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.activation = activation

    def gen_params(self):
        size = (
            self.hidden_size * (self.feature_size + self.hidden_size2)
            + self.hidden_size2
        )
        return torch.rand(size) * 2 - 1

    def forward(self, params, features):
        num_hidden_features = self.feature_size * self.hidden_size
        num_hidden_features2 = self.hidden_size * self.hidden_size2
        num_hidden_features_all = num_hidden_features + num_hidden_features2

        hidden_params = params[:num_hidden_features].reshape(self.hidden_size, -1).t()
        hidden_params2 = (
            params[num_hidden_features:num_hidden_features_all]
            .reshape(self.hidden_size2, -1)
            .t()
        )

        res = self.activation(features @ hidden_params)
        res2 = self.activation(res @ hidden_params2)
        return params[num_hidden_features_all:].dot(res2)


class CollaborativeFilteringRecommender(BaseRanker):
    def forward(self, user_embedding, item_embeddings):
        return user_embedding.reshape(1, -1) @ item_embeddings.t()

    def federated_item_grad(
        self, user_embedding, item_embeddings, interactions, alpha=0
    ):
        user_embedding = user_embedding.reshape(1, -1)
        interactions = interactions.reshape(1, -1)
        confidence = 1 + alpha * interactions
        fx = self.forward(user_embedding, item_embeddings)
        return (confidence * (interactions - fx)).t() @ user_embedding


class NeuralCollaborativeFilteringRecommender(nn.Module):
    def __init__(self, embedding_size, hidden_sizes):
        super().__init__()
        self.first_layer = nn.Linear(embedding_size * 2, hidden_sizes[0])
        self.layers = []
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.final_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, user_embedding, item_embeddings):
        embeddings = torch.cat(
            [user_embedding.expand(item_embeddings.shape[0], -1), item_embeddings], dim=1
        )
        res = F.relu(self.first_layer(embeddings))
        for layer in self.layers:
            res = F.relu(layer(res))
        return F.sigmoid(self.final_layer(res))

    def item_grad(self, user_embedding, item_embeddings, interactions):
        for p in self.parameters():
            p.requires_grad = False
        def f(item_embeddings):
            preds = self.forward(user_embedding, item_embeddings)
            loss = F.binary_cross_entropy(preds.view(-1), interactions)
            return loss
        return grad(f)(item_embeddings)
    
    def feature_grad(self, user_embedding, item_embeddings, interactions, retain_graph=False):
        for p in self.parameters():
            p.requires_grad = True
            if p.grad is not None:
                p.grad.zero_()
        
        preds = self.forward(user_embedding, item_embeddings)
        loss = F.binary_cross_entropy(preds.view(-1), interactions)
        loss.backward(retain_graph=retain_graph)
        
        feature_grads = []
        for p in self.parameters():
            feature_grads.append(p.grad.clone().flatten())
            p.grad.zero_()
            p.requires_grad = False
        
        return torch.cat(feature_grads)

if __name__ == "__main__":
    num_features = 5
    num_data = 3
    X = torch.rand(num_data, num_features)
    ranking = list(range(num_data))
    random.shuffle(ranking)
    ranking = torch.LongTensor(ranking)
    interactions = torch.randint(0, 2, (num_data,))
    while interactions.sum() == 0:
        interactions = torch.randint(0, 2, (num_data,))
    user_embedding = torch.rand(num_features)

    # ranker = LinearPDGDRanker(num_features)
    # print(ranker.grad(torch.rand(num_features), X, ranking, interactions))

    # hidden_size = 2
    # ranker2 = Neural1LayerPDGDRanker(num_features, hidden_size)
    # print(
    #     ranker2.grad(
    #         torch.rand((num_features + 1) * hidden_size), X, ranking, interactions
    #     )
    # )

    # hidden_size2 = 2
    # ranker3 = Neural2LayerPDGDRanker(num_features, hidden_size, hidden_size2)
    # print(
    #     ranker3.grad(
    #         torch.rand(
    #             num_features * hidden_size + hidden_size * hidden_size2 + hidden_size2
    #         ),
    #         X,
    #         ranking,
    #         interactions,
    #     )
    # )
    
    # cf_rec = CollaborativeFilteringRecommender()
    # print(cf_rec.federated_item_grad(user_embedding, X, interactions))

    ncf_rec = NeuralCollaborativeFilteringRecommender(num_features, [2])
    print(ncf_rec.item_grad(user_embedding, X, interactions.float()))
    print(ncf_rec.feature_grad(user_embedding, X, interactions.float()))
