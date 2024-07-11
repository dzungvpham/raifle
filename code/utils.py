import json
import math
import numpy as np
import pandas as pd
import torch
from dataset import (
    LearningToRankDataset,
)
from diffprivlib.mechanisms import (
    Gaussian,
    GaussianAnalytic,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Metrics:
    def __init__(self, path=None):
        if path is not None:
            self.df = pd.read_csv(path)
        else:
            self.df = pd.DataFrame(
                {
                    "name": [],
                    "accuracy": [],
                    "f1": [],
                    "precision": [],
                    "recall": [],
                    "auc": [],
                    "auc-pr": [],
                    "extra_data": [],
                }
            )

    def update(self, name, target, preds, preds_raw=None, extra_data={}):
        row = {
            "name": name,
            "accuracy": accuracy_score(target, preds),
            "f1": f1_score(target, preds),
            "precision": precision_score(target, preds, zero_division=0),
            "recall": recall_score(target, preds),
            "auc": None if preds_raw is None else roc_auc_score(target, preds_raw),
            "auc-pr": None
            if preds_raw is None
            else average_precision_score(target, preds_raw),
            "extra_data": json.dumps(extra_data),
        }
        self.df.loc[len(self.df.index), :] = row

    def get_dataframe(self):
        return self.df

    def save(self, path):
        self.df.to_csv(path, index=False)

    def load(self, path):
        self.df = pd.read_csv(path)

    def print_summary(self, metrics=["auc"]):
        print(self.df[["name"] + metrics].groupby("name").describe().to_string())


class ClickModel:
    def __init__(self):
        raise NotImplementedError

    def click(self):
        raise NotImplementedError


class CascadeClickModel(ClickModel):
    def __init__(self, prob_click, prob_stop):
        self.prob_click = prob_click
        self.prob_stop = prob_stop

    def click(self, ranking, relevance, filter_all_or_zero=True):
        n = len(ranking)
        clicks = [False] * n
        while True:
            for i in range(n):
                r = relevance[ranking[i]]
                clicks[i] = np.random.rand() < self.prob_click[r]
                if clicks[i] and np.random.rand() < self.prob_stop[r]:
                    break
            
            if (not filter_all_or_zero or (np.any(clicks) and not np.all(clicks))):
                break

        return clicks
    
class LtrEvaluator():
    def __init__(self, dataset: LearningToRankDataset, rank_cnt: int):
        self._dataset = dataset
        self._qids = dataset.get_all_query_ids()
        self._rank_cnt = rank_cnt
        self._idcg_map = {}
        self._dcg_weight = 1/np.log2(np.arange(2, rank_cnt + 2))

        # Pre-populate ideal DCG for all queries
        for qid in self._qids:
            relevance = dataset.get_data_for_queries([qid])[0][0].copy()
            relevance.sort()  # Ascending
            # Reverse and grab top k
            relevance = relevance[:-(self._rank_cnt+1):-1]
            self._idcg_map[qid] = np.sum(
                (2 ** np.array(relevance) - 1) * self._dcg_weight[:len(relevance)])

    def calculate_average_offline_ndcg(self, model, params) -> float:
        total_dcg = 0.0
        for qid in self._qids:
            if self._idcg_map[qid] == 0.0:
                continue

            relevance, features = self._dataset.get_data_for_queries([qid])[0]
            ranking = model.rank(params, torch.Tensor(features), sample=False)
            total_dcg += self.calculate_ndcg_for_query_ranking(
                qid, ranking, np.array(relevance))

        return total_dcg / len(self._qids)
    
    def calculate_ndcg_for_query_ranking(self, qid, ranking, relevance) -> float:
        idcg = self._idcg_map[qid]
        ranking = ranking[:self._rank_cnt]
        ranking_relevance = relevance[ranking]
        n = len(ranking)
        dcg = np.sum((2 ** ranking_relevance - 1) * self._dcg_weight[:n])
        return dcg / idcg

# Clip and add Gaussian noise to a torch tensor
def apply_gaussian_mechanism(input, epsilon, delta, sensitivity, scale_only=False):
    if math.isinf(epsilon):
        return input
    # Clip L2 norm to 0.5 * sensitivity (since global L2 sensitivity = 2 * max L2 norm)
    output = input * torch.minimum(torch.tensor(1.0), 0.5 * sensitivity / torch.linalg.vector_norm(input))
    if scale_only:
        return output

    # Add noise
    mechanism = (Gaussian if epsilon <= 1.0 else GaussianAnalytic)(
        epsilon=epsilon, delta=delta, sensitivity=sensitivity
    )
    return output.apply_(mechanism.randomise)
