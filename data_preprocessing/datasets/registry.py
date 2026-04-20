from data_preprocessing.datasets.vision import (
    download_mnist,
    download_cifar10,
    download_fashion_mnist,
)


DATASET_REGISTRY = {
    "mnist": download_mnist,
    "cifar10": download_cifar10,
    "fmnist": download_fashion_mnist,
}
