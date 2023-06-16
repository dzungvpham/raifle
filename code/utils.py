import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

class Metrics:
    def __init__(self):
        self.df = pd.DataFrame({
            'name': [],
            'accuracy': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'auc': [],
            'auc-pr': [],
            'extra_data': [],
        })

    def update(self, name, target, preds, preds_raw=None, extra_data={}):
        row = {
            'name': name,
            'accuracy': accuracy_score(target, preds),
            'f1': f1_score(target, preds),
            'precision': precision_score(target, preds, zero_division=0),
            'recall': recall_score(target, preds),
            'auc': None if preds_raw is None else roc_auc_score(target, preds_raw),
            'auc-pr': None if preds_raw is None else average_precision_score(target, preds_raw),
            'extra_data': json.dumps(extra_data),
        }
        self.df.loc[len(self.df.index), :] = row

    def get_dataframe(self):
        return self.df

    def save(self, path):
        self.df.to_csv(path, index=False)

    def load(self, path):
        self.df = pd.read_csv(path)

class ClickModel():
    def __init__(self):
        raise NotImplementedError
    
    def click(self):
        raise NotImplementedError

class CascadeClickModel(ClickModel):
    def __init__(self, prob_click, prob_stop):
        self.prob_click = prob_click
        self.prob_stop = prob_stop

    def click(self, ranking, relevance):
        n = len(ranking)
        clicks = [False] * n
        while not np.any(clicks):
            for i in range(n):
                r = relevance[ranking[i]]
                clicks[i] = np.random.rand() < self.prob_click[r]
                if clicks[i] and np.random.rand() < self.prob_stop[r]:
                    break

        return clicks