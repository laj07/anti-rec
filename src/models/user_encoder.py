import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        ub = self.user_bias(user).squeeze(1)
        ib = self.item_bias(item).squeeze(1)
        return (u * i).sum(dim=1) + ub + ib

    def get_user_vector(self, user_id):
        idx = torch.tensor([user_id], dtype=torch.long)
        return self.user_emb(idx).detach().squeeze(0)

    def get_all_item_vectors(self):
        return self.item_emb.weight.detach()
