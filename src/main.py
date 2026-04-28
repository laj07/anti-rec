import argparse
import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from data.dataset import RatingsDataset
from models.user_encoder import MatrixFactorization
from models.anti_rec import AntiRecommender
from train import train
from evaluate import evaluate


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["train", "eval", "demo"], default="train")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RatingsDataset(cfg["data"])

    model = MatrixFactorization(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        dim=cfg["model"]["dim"],
    ).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.mode == "train":
        train(model, dataset, cfg["training"], device)

    elif args.mode == "eval":
        anti_rec = AntiRecommender(model, cfg["model"], dataset.popularity)
        results = evaluate(anti_rec, dataset, cfg["eval"])
        print("\n=== Evaluation Results ===")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    elif args.mode == "demo":
        anti_rec = AntiRecommender(model, cfg["model"], dataset.popularity)
        user_id = int(input("Enter user ID (0 to {}): ".format(dataset.n_users - 1)))
        alpha = float(input("Anti-recommendation strength (0.0=normal, 1.0=max surprise): "))
        seen = dataset.user_history.get(user_id, [])
        recs = anti_rec.recommend(user_id, alpha=alpha, k=10, exclude_seen=seen)
        print(f"\nTop-10 anti-recommendations (alpha={alpha}):")
        for rank, (item_id, score) in enumerate(recs, 1):
            title = dataset.item_titles.get(item_id, f"item_{item_id}")
            print(f"  {rank:2d}. {title}  (score: {score:.3f})")


if __name__ == "__main__":
    main()
