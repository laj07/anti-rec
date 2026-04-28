from models.user_encoder import UserEncoder
from models.anti_rec import AntiRecommender
from data.dataset import DummyDataset

def main():
    dataset = DummyDataset()

    user = dataset.get_user(0)
    items = dataset.get_items()

    encoder = UserEncoder(20, 16)
    anti = AntiRecommender()

    user_emb = encoder(user)
    ranked = anti.generate(user_emb, items)

    print("Top anti recommendations:", ranked[:5])

if __name__ == "__main__":
    main()