import torch

class DummyDataset:
    def __init__(self):
        self.users = torch.randn(100, 20)
        self.items = torch.randn(500, 16)

    def get_user(self, idx):
        return self.users[idx]

    def get_items(self):
        return self.items