# Anti-Recommendation System

> Deliberately surfaces content outside your comfort zone — for exposure therapy, debate prep, and filter bubble escape.

## How It Works

Train a preference model on user ratings, then **invert it**. A single `alpha` dial controls intensity from normal recommendations (0.0) to maximum surprise (1.0).

## Example Output: User 42

**Normal recommendations (alpha=0.0):**
Happy, Texas (1999)
Sliding Doors (1998)
For Ever Mozart (1996)
Mighty, The (1998)
Girlfight (2000)
**Anti-recommendations (alpha=0.8):**
Star Wars: Episode V - The Empire Strikes Back (1980)
Pulp Fiction (1994)
Fargo (1996)
Godfather, The (1972)
Annie Hall (1977)
Citizen Kane (1941)
User 42 watches light indie/feel-good films. The model pushes them toward dense, critically-acclaimed classics — a genuine taste boundary.

## Three Strategies

| Strategy | Description |
|---|---|
| `boundary` | Items at the edge of your preference cluster — surprising but real |
| `negation` | Flip the dot-product score sign |
| `adversarial` | Iteratively perturb user embedding |

## Quick Start

```bash
pip install -r requirements.txt

wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/ && rm ml-1m.zip

python src/main.py --config experiments/movies/config_1m.yaml --mode train
python src/main.py --config experiments/movies/config_1m.yaml --mode demo --checkpoint outputs/best_model.pt
```

## Results

Trained on MovieLens 1M — 6040 users, 3706 movies, 1M ratings.
Best validation loss: **0.0042** after 10 epochs on CPU.

