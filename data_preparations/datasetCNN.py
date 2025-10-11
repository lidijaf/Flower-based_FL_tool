import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from typing import List, Tuple
from utils import power_law_split, sort_by_class
from utils import get_cfg


def download_dataset(dataset_name: str, data_path: str) -> Tuple[Dataset, Dataset]:
    """Download dataset and apply minimal transformation."""
    if dataset_name == 'mnist':
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        trainset = MNIST(data_path, train=True, download=True, transform=transform)
        testset = MNIST(data_path, train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = CIFAR10(data_path, train=True, download=True, transform=transform)
        testset = CIFAR10(data_path, train=False, download=True, transform=transform)
    elif dataset_name == "fmnist":
        transform = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])
        trainset = FashionMNIST(data_path, train=True, download=True, transform=transform)
        testset = FashionMNIST(data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return trainset, testset


def load_datasets(cfg) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    #cfg = get_cfg()
    dataset_name = cfg.get("dataset")
    num_partitions = cfg.get("num_clients")
    batch_size = cfg.get("batch_size")
    val_ratio = cfg.get("val_ratio")
    iid_partition = cfg.get("iid_partition")
    seed = cfg.get("seed")
    num_workers = cfg.get("num_workers")


    """Download dataset and generate either IID or non-IID partitions based on params."""

    # Download dataset
    path = "C://Users//neman//PycharmProjects//data"
    trainset, testset = download_dataset(dataset_name, path)

    # Create partitions
    if iid_partition:
        partition_len = [len(trainset) // num_partitions] * num_partitions
        trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(seed))
    else:
        trainset_sorted = sort_by_class(trainset)
        trainsets = power_law_split(trainset_sorted)

    # Create loaders
    trainloaders, valloaders = [], []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(seed))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=num_workers))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=num_workers))

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloaders, valloaders, testloader

import os
from torch.utils.data import DataLoader




def load_data(cfg) -> Tuple[DataLoader, DataLoader]:
    #cfg = get_cfg()
    data_path = cfg["data_path"]

    train_data = torch.load(os.path.join(data_path, "train.pt"))
    test_data = torch.load(os.path.join(data_path, "test.pt"))

    trainloader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
    testloader = DataLoader(test_data, batch_size=cfg["batch_size"], shuffle=False)

    return trainloader, testloader
