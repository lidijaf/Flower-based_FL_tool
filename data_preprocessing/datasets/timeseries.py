import torch
import numpy as np
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    def __init__(self, data, labels, win_size, step, mode="train"):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.win_size = win_size
        self.step = step
        self.mode = mode

    def __len__(self):
        return (self.data.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step

        x = self.data[index:index + self.win_size]

        if self.mode == "test":
            y = self.labels[index:index + self.win_size]
        else:
            y = self.labels[:self.win_size]

        return np.float32(x), np.float32(y)
