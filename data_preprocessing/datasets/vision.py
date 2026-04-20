import os
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose


def download_mnist(data_dir):
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = MNIST(data_dir, train=True, download=True, transform=transform)
    test = MNIST(data_dir, train=False, download=True, transform=transform)
    return train, test


def download_cifar10(data_dir):
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = CIFAR10(data_dir, train=True, download=True, transform=transform)
    test = CIFAR10(data_dir, train=False, download=True, transform=transform)
    return train, test


def download_fashion_mnist(data_dir):
    transform = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])
    train = FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test = FashionMNIST(data_dir, train=False, download=True, transform=transform)
    return train, test
