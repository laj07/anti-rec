import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class RatingsDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        path = cfg["path"]
        sep = cfg.get("sep", ",")
        min_ratings = cfg.get("min_ratings_per_user", 20)

        print(f"Loading ratings from {path}...")
        df = pd.read_csv(path, sep=sep, header=None, engine="python",
                         names=["userId","movieId","rating","timestamp"])

        counts = df.groupby("userId")["rating"].count()
        active_users = counts[counts >= min_ratings].index
        df = df[df["userId"].isin(active_users)]

        self.user2idx = {u: i for i, u in enumerate(df["userId"].unique())}
        self.item2idx = {m: i for i, m in enumerate(df["movieId"].unique())}
        self.idx2item = {i: m for m, i in self.item2idx.items()}

        df["user"] = df["userId"].map(self.user2idx)
        df["item"] = df["movieId"].map(self.item2idx)

        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)
        print(f"  {self.n_users} users, {self.n_items} items, {len(df)} ratings")

        df["rating_norm"] = df["rating"] / df["rating"].max()

        df = df.sort_values("timestamp")
        val_size = int(len(df) * cfg.get("val_split", 0.1))
        self.val_df = df.tail(val_size).reset_index(drop=True)
        self.train_df = df.iloc[:-val_size].reset_index(drop=True)

        self.user_history = (
            self.train_df.groupby("user")["item"].apply(list).to_dict()
        )

        item_counts = self.train_df.groupby("item")["rating"].count()
        pop = np.zeros(self.n_items)
        for item_idx, cnt in item_counts.items():
            pop[item_idx] = cnt
        pop = np.log1p(pop)
        pop = pop / pop.max()
        self.popularity = torch.tensor(pop, dtype=torch.float32)

        self.item_titles = {}
        titles_path = cfg.get("titles_path")
        if titles_path and os.path.exists(titles_path):
            self._load_titles(titles_path, sep)

        self._active_df = self.train_df

    def use_val(self):
        self._active_df = self.val_df

    def use_train(self):
        self._active_df = self.train_df

    def __len__(self):
        return len(self._active_df)

    def __getitem__(self, idx):
        row = self._active_df.iloc[idx]
        return (
            torch.tensor(int(row["user"]), dtype=torch.long),
            torch.tensor(int(row["item"]), dtype=torch.long),
            torch.tensor(float(row["rating_norm"]), dtype=torch.float32),
        )

    def _load_titles(self, path, sep=","):
        movies = pd.read_csv(path, sep=sep, header=None, engine="python",
                             names=["movieId","title","genres"],
                             encoding="latin-1")
        for _, row in movies.iterrows():
            if row["movieId"] in self.item2idx:
                self.item_titles[self.item2idx[row["movieId"]]] = row["title"]
        print(f"  Loaded {len(self.item_titles)} titles")
