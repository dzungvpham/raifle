import torch
import torch.nn as nn
from torch.func import grad, vmap

class BaseRanker:
    def __init__(self):
        pass
        
    def forward(self):
        raise NotImplementedError
        
    def grad(self):
        raise NotImplementedError
    
    def fit(self):
        raise NotImplementedError
    
class BasePDGDRanker(BaseRanker):        
    def forward(self, params, features):
        raise NotImplementedError
    
    def grad(self, params, features, ranking, interactions):
        num_items = features.shape[0]
        inputs = (params, features)
        fx = vmap(self.forward, in_dims=(None, 0))(*inputs)
        
        # Calculate position bias reweighing
        log_prob_orig = 0.5
        log_prob_swapped = 0.5
        
        log_pos_bias_weight = log_prob_swapped - torch.logsumexp(torch.Tensor([log_prob_orig, log_prob_swapped]), 0)
        
        # Calculate gradient of probabilities of click pairs
        
        fx_expanded = fx.expand(num_items, -1)
        fx_sum = fx_expanded.t() + fx_expanded
        fx_logsumexp = torch.logsumexp(torch.cartesian_prod(fx, fx), dim=1).reshape(num_items, num_items)
        
        fx_grad = vmap(grad(self.forward), in_dims=(None, 0))(*inputs)
        fx_grad_diff = vmap(torch.sub, in_dims=(0, None))(*(fx_grad, fx_grad))
        
        # Calculate final gradients
        
        interaction_matrix = interactions @ (1 - interactions.t())
        weights = (log_pos_bias_weight + fx_sum - 2 * fx_logsumexp).exp()        
        res = interaction_matrix.unsqueeze(2) * weights.unsqueeze(2) * fx_grad_diff
        
        return res.sum(dim=(0,1))
    
class LinearPDGDRanker(BasePDGDRanker):
    def forward(self, params, features):
        return params.dot(features)
    
if __name__ == "__main__":
    ranker = LinearPDGDRanker()
    print(ranker.grad(torch.rand(5), torch.rand(3, 5), torch.Tensor([0, 1, 2]), torch.Tensor([1, 0, 1]).reshape(-1, 1)))