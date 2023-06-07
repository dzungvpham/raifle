import random
import torch
from attack import (
    reconstruct_interactions,
    reconstruct_interactions_with_private_params,
)
from ranker import (
    LinearPDGDRanker,
    Neural1LayerPDGDRanker,
    Neural2LayerPDGDRanker,
    CollaborativeFilteringRecommender,
)

if __name__ == "__main__":
    num_features = 100
    num_data = 1000
    features = torch.rand(num_data, num_features)

    interactions = torch.randint(0, 2, (num_data,))
    # interactions = torch.bernoulli(torch.ones(num_data) * 0.2)
    print(interactions.long())

    # ranking = list(range(num_data))
    # random.shuffle(ranking)
    # ranking = torch.LongTensor(ranking)
    # lr = 0.1

    # params = torch.rand(num_features)
    # ranker = LinearPDGDRanker()

    # hidden_size = 5
    # params2 = torch.rand((num_features + 1) * hidden_size)
    # ranker2 = Neural1LayerPDGDRanker(num_features, hidden_size)

    # hidden_size2 = 2
    # params3 = torch.rand(
    #     num_features * hidden_size + hidden_size * hidden_size2 + hidden_size2
    # )
    # ranker3 = Neural2LayerPDGDRanker(num_features, hidden_size, hidden_size2)

    # def train(params, model):
    #     return lambda interactions: params + lr * model.grad(
    #         params, features, ranking, interactions
    #     )

    # train1 = train(params, ranker)
    # train2 = train(params2, ranker2)
    # train3 = train(params3, ranker3)
    
    # print(
    #     reconstruct_interactions(
    #         train1, train1(interactions), num_data, lr=0.001, max_iters=1000
    #     )
    # )
    # print(
    #     reconstruct_interactions(
    #         train2, train2(interactions), num_data, lr=0.01, max_iters=1000
    #     )
    # )
    # print(
    #     reconstruct_interactions(
    #         train3, train3(interactions), num_data, lr=0.01, max_iters=1000
    #     )
    # )

    cf_rec = CollaborativeFilteringRecommender()
    user_embedding = torch.rand(num_features)
    print(user_embedding)

    user_embedding2 = torch.rand(num_features)
    target = cf_rec.federated_item_grad(user_embedding, features, interactions)

    def train_cf(interactions):
        return cf_rec.federated_item_grad(
            user_embedding2, features, interactions
        )

    def train_cf2(interactions, private_user_embedding):
        return cf_rec.federated_item_grad(
            private_user_embedding, features, interactions
        )

    # print(
    #     reconstruct_interactions(
    #         train_cf, target, num_data, lr=0.001, max_iters=100000
    #     )
    # )

    print(
        reconstruct_interactions_with_private_params(
            train_cf2,
            target,
            num_data,
            num_features,
            lr=0.001,
            max_iters=100000,
            num_rounds=10,
        )
    )
