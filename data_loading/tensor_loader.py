import os

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


def _as_dataset(data):
    if isinstance(data, Dataset):
        return data

    if isinstance(data, tuple):
        return TensorDataset(*data)

    if isinstance(data, list):
        return TensorDataset(*data)

    if torch.is_tensor(data):
        dummy_labels = torch.zeros(len(data), dtype=torch.long)
        return TensorDataset(data, dummy_labels)

    raise TypeError(f"Unsupported dataset type: {type(data)}")

def load_tensor_splits(cfg):
    data_path = cfg["data_path"]
    batch_size = cfg["batch_size"]

    train_data = torch.load(os.path.join(data_path, "train.pt"))
    test_data = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = _as_dataset(train_data)
    test_dataset = _as_dataset(test_data)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
