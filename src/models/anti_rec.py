import torch

class AntiRecommender:
    def generate(self, user_emb, item_embs, alpha=0.7):
        scores = item_embs @ user_emb
        anti_scores = -alpha * scores
        return torch.argsort(anti_scores, descending=True)