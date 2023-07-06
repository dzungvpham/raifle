import numpy as np
import os.path
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler


class LearningToRankDataset:
    def __init__(self, path="", normalize=False):
        print(f"Loading {path}")

        rows = []
        with open(path, "r") as file:
            for line in tqdm(file):
                line_parts = line.split(" ")
                row = {
                    "qid": int(line_parts[1].split(":")[1]),
                    "relevance": int(line_parts[0]),
                }

                for col in line_parts[2:]:
                    if col == "#docid":
                        break
                    if ":" not in col:
                        continue

                    feature, feature_val = col.split(":")
                    row[int(feature)] = float(feature_val)

                rows.append(row)

        df = pd.DataFrame(rows)

        if normalize:
            print("Normalizing...")
            normalized = MinMaxScaler().fit_transform(df.loc[:, ~df.columns.isin(["qid", "relevance"])])
            df[df.loc[:, ~df.columns.isin(["qid", "relevance"])].columns] = normalized.astype('float32')

        print("Processing...")
        self.qids = df["qid"].unique().tolist()
        self.qid_to_data_map = df.groupby("qid").apply(
            lambda x: (x["relevance"].values.tolist(), x.iloc[:, 2:].values.tolist())
        )
        self.num_features = len(df.columns) - 2

    def get_all_query_ids(self) -> List:
        return self.qids
    
    # Returns an array of tuple of the form (relevances, features)
    def get_data_for_queries(self, qids: List) -> List[Tuple[List, List]]:
        return self.qid_to_data_map.loc[qids].tolist()

    def get_num_features(self) -> int:
        return self.num_features


class RecommendationDataset:
    def __init__(self, df, user_id_col, item_id_col, rating_col):
        self.user_ids = np.sort(df[user_id_col].unique()).tolist()
        self.item_ids = np.sort(df[item_id_col].unique()).tolist()
        self.user_to_item_rating_map = df.groupby(user_id_col).apply(
            lambda x: dict(zip(x[item_id_col], x[rating_col]))
        )

    def get_all_user_ids(self) -> List:
        return self.user_ids

    def get_all_item_ids(self) -> List:
        return self.item_ids

    def get_item_ids_for_users(self, user_ids: List) -> List[List]:
        return (
            self.user_to_item_rating_map.loc[user_ids]
            .apply(lambda x: list(x.keys()))
            .tolist()
        )

    def get_non_interacted_item_ids_for_users(self, user_ids: List) -> List[List]:
        return (
            self.user_to_item_rating_map.loc[user_ids]
            .apply(lambda x: list(set(self.get_all_item_ids()) - set(x.keys())))
            .tolist()
        )

    # Return a list of dictionary from item id to rating
    def get_item_ratings_for_users(self, user_ids: List) -> List[Dict]:
        return self.user_to_item_rating_map.loc[user_ids].tolist()


class MovieLens(RecommendationDataset):
    def __init__(self, path="../dataset/ML-100K/u.data", sep="\t"):
        df = pd.read_csv(
            path,
            sep=sep,
            header=None,
            names=["user_id", "item_id", "rating"],
            usecols=["user_id", "item_id", "rating"],
        )
        super().__init__(df, "user_id", "item_id", "rating")


class Goodreads(RecommendationDataset):
    def __init__(self, path="../dataset/Goodreads/goodreads_interactions.csv"):
        df = pd.read_csv(
            path,
            usecols=["user_id", "book_id", "rating"],
            engine="c",
            low_memory=True,
            dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.int32},
        )
        super().__init__(df, "user_id", "book_id", "rating")


class Yelp(RecommendationDataset):
    def __init__(
        self,
        path="../dataset/Yelp/yelp_academic_dataset_review.json",
        cache_path="../dataset/Yelp/review.csv",
    ):
        if cache_path is not None and os.path.isfile(cache_path):
            df = pd.read_csv(cache_path)
        else:  # Preprocess
            print(f"Processing {path}")
            reader = pd.read_json(path, lines=True, chunksize=10000)
            df = pd.DataFrame()
            for chunk in tqdm(reader):
                df = pd.concat([df, chunk[["user_id", "business_id", "stars"]]])
            df["user_id"], _ = pd.factorize(df["user_id"])
            df["business_id"], _ = pd.factorize(df["business_id"])
            df.to_csv(cache_path, index=False)

        super().__init__(df, "user_id", "business_id", "stars")
