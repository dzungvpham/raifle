import numpy as np
import os.path
import pandas as pd


class RecommendationDataset:
    def __init__(self, df, user_id_col, item_id_col, rating_col):
        self.user_ids = np.sort(df[user_id_col].unique()).tolist()
        self.item_ids = np.sort(df[item_id_col].unique()).tolist()
        self.user_to_item_rating_map = df.groupby(user_id_col).apply(
            lambda x: dict(zip(x[item_id_col], x[rating_col]))
        )

    def get_all_user_ids(self) -> list:
        return self.user_ids

    def get_num_users(self):
        return len(self.user_ids)

    def get_all_item_ids(self) -> list:
        return self.item_ids

    def get_num_items(self):
        return len(self.item_ids)

    def get_item_ids_for_users(self, user_ids: list):
        return (
            self.user_to_item_rating_map.loc[user_ids]
            .apply(lambda x: list(x.keys()))
            .tolist()
        )

    def get_non_interacted_item_ids_for_users(self, user_ids: list):
        return (
            self.user_to_item_rating_map.loc[user_ids]
            .apply(lambda x: list(set(self.get_all_item_ids()) - set(x.keys())))
            .tolist()
        )

    def get_item_ratings_for_users(self, user_ids: list):
        return self.user_to_item_rating_map.loc[user_ids].tolist()


class MovieLens(RecommendationDataset):
    def __init__(self, path="../dataset/ML-100K/u.data"):
        df = pd.read_csv(
            path,
            sep="\t",
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
            reader = pd.read_json(path, lines=True, chunksize=10000)
            df = pd.DataFrame()
            for chunk in reader:
                df = pd.concat([df, chunk[["user_id", "business_id", "stars"]]])
            df["user_id"], _ = pd.factorize(df["user_id"])
            df["business_id"], _ = pd.factorize(df["business_id"])
            df.to_csv(cache_path, index=False)

        super().__init__(df, "user_id", "business_id", "stars")
