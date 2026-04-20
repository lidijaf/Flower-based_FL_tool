import os
import torch
from torch.utils.data import DataLoader


def load_data(cfg):
    data_path = cfg["data_path"]
    batch_size = cfg["batch_size"]

    train_data = torch.load(os.path.join(data_path, "train.pt"))
    test_data = torch.load(os.path.join(data_path, "test.pt"))

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
