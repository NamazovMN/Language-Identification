import torch
from torch.utils.data import Dataset


class LangID(Dataset):
    def __init__(self, dataset, device):
        self.device = device
        self.data, self.label = self.get_dataset(dataset)

    def get_dataset(self, dataset):
        data = list(dataset['encoded'])
        labels = list(dataset['label'])
        return torch.LongTensor(data).to(self.device), torch.LongTensor(labels).to(self.device)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'label': self.label[idx]
        }

    def __len__(self):
        return len(self.label)
