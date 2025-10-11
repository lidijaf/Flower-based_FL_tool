import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from typing import List, Tuple
from utils import get_cfg


# TRAINING DATASET
nonanomaly_path = 'C://Users//neman//PycharmProjects//data//METRO//metro_nonanomaly//metro_nonanomaly.csv'
nonanomaly_data = pd.read_csv(nonanomaly_path).values
train_data_matrix = nonanomaly_data[:16000, :-1]  # All rows until the last column
train_labels = nonanomaly_data[:16000, -1]  # Only the last column

# TESTING DATASET
anomaly_path = 'C://Users//neman//PycharmProjects//data//METRO//metro_anomaly//metro_anomaly.csv'
anomaly_data = pd.read_csv(anomaly_path).values
test_data_matrix_pt1 = anomaly_data[:1000, :-1]  # All rows until the last column
test_labels_pt1 = anomaly_data[:1000, -1]  # Only the last column

# Add the same amount of non-anomalies to the testing dataset
test_data_matrix_pt2 = nonanomaly_data[16000:17000, :-1]
test_labels_pt2 = nonanomaly_data[16000:17000, -1]

# Combine test data vertically (vstack for data matrix, hstack for labels)
test_data_matrix = np.vstack((test_data_matrix_pt1, test_data_matrix_pt2))
test_labels = np.hstack((test_labels_pt1, test_labels_pt2))


# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)   # Convert data to torch tensor
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to torch tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        label = self.labels[idx]
        return data_sample, label


# Create an instance of the dataset
train_dataset = CustomDataset(train_data_matrix, train_labels)

# Create an instance of the dataset
test_dataset = CustomDataset(test_data_matrix, test_labels)


def load_datasets() -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    cfg = get_cfg()
    num_partitions = cfg.get("num_clients")
    batch_size = cfg.get("batch_size")
    val_ratio = cfg.get("val_ratio")
    seed = cfg.get("seed")
    num_workers = cfg.get("num_workers")

    partition_len = [len(train_dataset) // num_partitions] * num_partitions
    trainsets = random_split(train_dataset, partition_len, torch.Generator().manual_seed(seed))

    trainloaders, valloaders = [], []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(seed))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=num_workers))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=num_workers))

    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloaders, valloaders, testloader
