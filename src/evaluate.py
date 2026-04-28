import torch
import torch.nn.functional as F
import numpy as np


def evaluate(anti_rec, dataset, cfg):
    metrics = cfg.get("metrics", ["embedding_surprise", "novelty"])
    k = cfg.get("k", 10)
    n_users = cfg.get("n_eval_users", 100)
    alpha = cfg.get("alpha", 0.8)

    eligible = [u for u, h in dataset.user_history.items() if len(h) >= 10]
    rng = np.random.default_rng(42)
    sampled = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)

    scores = {m: [] for m in metrics}
    for user_id in sampled:
        seen = dataset.user_history.get(user_id, [])
        recs = anti_rec.recommend(user_id, alpha=alpha, k=k, exclude_seen=seen)
        rec_items = [r[0] for r in recs]

        if "embedding_surprise" in metrics:
            scores["embedding_surprise"].append(
                _embedding_surprise(anti_rec.model, user_id, rec_items))
        if "novelty" in metrics:
            scores["novelty"].append(
                _novelty(dataset.popularity, rec_items))
        if "intra_list_diversity" in metrics:
            scores["intra_list_diversity"].append(
                _intra_list_diversity(anti_rec._item_vecs, rec_items))

    return {m: float(np.mean(v)) for m, v in scores.items() if v}


def _embedding_surprise(model, user_id, rec_items):
    user_vec = model.get_user_vector(user_id)
    item_vecs = model.get_all_item_vectors()[rec_items]
    user_norm = F.normalize(user_vec.unsqueeze(0), dim=1)
    item_norms = F.normalize(item_vecs, dim=1)
    return float((1.0 - (item_norms @ user_norm.T).squeeze(1)).mean())


def _novelty(popularity, rec_items):
    if popularity is None:
        return 0.0
    return float(1.0 - popularity[rec_items].mean())


def _intra_list_diversity(item_vecs, rec_items):
    if len(rec_items) < 2:
        return 0.0
    vecs = F.normalize(item_vecs[rec_items], dim=1)
    sim = vecs @ vecs.T
    n = len(rec_items)
    mask = ~torch.eye(n, dtype=torch.bool)
    return float(1.0 - sim[mask].mean())
