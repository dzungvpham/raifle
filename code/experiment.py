import random
import torch
from attack import reconstruct_interactions
from ranker import LinearPDGDRanker, Neural1LayerPDGDRanker, Neural2LayerPDGDRanker

if __name__ == "__main__":
    num_features = 10
    num_data = 30
    features = torch.rand(num_data, num_features)
    ranking = list(range(num_data))
    random.shuffle(ranking)
    ranking = torch.LongTensor(ranking)
    interactions = torch.randint(0, 2, (1, num_data))
    lr = 0.1
    
    params = torch.rand(num_features)
    ranker = LinearPDGDRanker()

    hidden_size = 5
    params2 = torch.rand((num_features + 1) * hidden_size)
    ranker2 = Neural1LayerPDGDRanker(num_features, hidden_size)

    hidden_size2 = 2
    params3 = torch.rand(num_features * hidden_size + hidden_size * hidden_size2 + hidden_size2)
    ranker3 = Neural2LayerPDGDRanker(num_features, hidden_size, hidden_size2)

    def train(params, model):
        return lambda interactions: params + lr * model.grad(params, features, ranking, interactions)

    train1 = train(params, ranker)
    train2 = train(params2, ranker2)
    train3 = train(params3, ranker3)

    print(interactions)
    print(reconstruct_interactions(train1, train1(interactions), num_data, lr=0.01, max_iters=1000))
    print(reconstruct_interactions(train2, train2(interactions), num_data, lr=0.01, max_iters=1000))
    print(reconstruct_interactions(train3, train3(interactions), num_data, lr=0.01, max_iters=1000))