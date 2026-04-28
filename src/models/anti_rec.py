import torch
import torch.nn.functional as F
import numpy as np


class AntiRecommender:
    def __init__(self, model, cfg, popularity=None):
        self.model = model
        self.model.eval()
        self.strategy = cfg.get("strategy", "boundary")
        self.popularity = popularity
        self._item_vecs = model.get_all_item_vectors()
        self._item_norms = F.normalize(self._item_vecs, dim=1)

    def recommend(self, user_id, alpha=0.8, k=10, exclude_seen=None):
        user_vec = self.model.get_user_vector(user_id)
        if self.strategy == "negation":
            scores = self._negation_scores(user_vec, alpha)
        elif self.strategy == "boundary":
            scores = self._boundary_scores(user_vec, alpha)
        elif self.strategy == "adversarial":
            scores = self._adversarial_scores(user_vec, alpha)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        if exclude_seen:
            scores[exclude_seen] = -1e9
        topk = torch.topk(scores, k=min(k, scores.shape[0]))
        return [(int(idx), float(score)) for idx, score in zip(topk.indices, topk.values)]

    def _negation_scores(self, user_vec, alpha):
        user_norm = F.normalize(user_vec.unsqueeze(0), dim=1)
        similarity = (self._item_norms @ user_norm.T).squeeze(1)
        return (1.0 - 2.0 * alpha) * similarity

    def _boundary_scores(self, user_vec, alpha):
        user_norm = F.normalize(user_vec.unsqueeze(0), dim=1)
        similarity = (self._item_norms @ user_norm.T).squeeze(1)
        if alpha == 0.0:
            return similarity
        target_pct = 1.0 - (alpha * 0.8)
        target_sim = float(np.percentile(similarity.numpy(), target_pct * 100))
        boundary_score = -(similarity - target_sim) ** 2
        if self.popularity is not None:
            pop = self.popularity.to(boundary_score.device)
            boundary_score = boundary_score + 0.1 * pop
        return boundary_score

    def _adversarial_scores(self, user_vec, alpha, steps=30, lr=0.05):
        if alpha == 0.0:
            user_norm = F.normalize(user_vec.unsqueeze(0), dim=1)
            return (self._item_norms @ user_norm.T).squeeze(1)
        perturbed = user_vec.clone().float()
        target = -user_vec
        for _ in range(steps):
            perturbed = perturbed + lr * alpha * (target - perturbed)
        perturbed_norm = F.normalize(perturbed.unsqueeze(0), dim=1)
        scores = (self._item_norms @ perturbed_norm.T).squeeze(1)
        original_norm = F.normalize(user_vec.unsqueeze(0), dim=1)
        original_scores = (self._item_norms @ original_norm.T).squeeze(1)
        return (1.0 - alpha) * original_scores + alpha * scores
