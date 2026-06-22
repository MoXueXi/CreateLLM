from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
