import math
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, Dataset, Subset


def get_cfg(relative_path: str):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, relative_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    return OmegaConf.load(config_path)


def sort_by_class(trainset: Dataset) -> Dataset:
    """Sort a dataset by class label."""

    if isinstance(trainset.targets, list):
        trainset.targets = torch.tensor(trainset.targets)

    idxs = trainset.targets.argsort()

    subsets = []
    subset_targets = []

    _, class_counts = torch.unique(trainset.targets, return_counts=True)
    start = 0

    for count in class_counts:
        end = start + count
        subsets.append(Subset(trainset, idxs[start:end]))
        subset_targets.append(trainset.targets[idxs[start:end]])
        start = end

    sorted_dataset = ConcatDataset(subsets)
    sorted_dataset.targets = torch.cat(subset_targets)

    return sorted_dataset


def power_law_split(sorted_trainset: Dataset) -> List[Dataset]:
    cfg = get_cfg()

    num_partitions = cfg.get("num_clients")
    num_labels_per_partition = cfg.get("num_labels_per_partition")
    min_data_per_partition = cfg.get("min_data_per_partition")
    mean = cfg.get("mean")
    sigma = cfg.get("sigma")

    required = {
        "num_clients": num_partitions,
        "num_labels_per_partition": num_labels_per_partition,
        "min_data_per_partition": min_data_per_partition,
        "mean": mean,
        "sigma": sigma,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required config values for power_law_split: {missing}")

    if isinstance(sorted_trainset.targets, list):
        sorted_trainset.targets = torch.tensor(sorted_trainset.targets)

    targets = sorted_trainset.targets
    full_idx = list(range(len(targets)))

    class_counts = torch.bincount(targets)
    labels_cs = torch.cumsum(class_counts, dim=0) - class_counts

    partitions_idx: List[List[int]] = []
    num_classes = len(class_counts)
    hist = torch.zeros(num_classes, dtype=torch.int32)

    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)

    for user_id in range(num_partitions):
        partitions_idx.append([])
        for class_idx in range(num_labels_per_partition):
            cls = (user_id + class_idx) % num_classes
            indices = full_idx[
                labels_cs[cls] : labels_cs[cls] + hist[cls] + min_data_per_class
            ]
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    safe_div = max(1, math.ceil(num_partitions / num_classes))
    probs = np.random.lognormal(
        mean, sigma, (num_classes, safe_div, num_labels_per_partition)
    )
    remaining_per_class = class_counts - hist

    probs = (
        remaining_per_class.reshape(-1, 1, 1).numpy()
        * probs
        / np.sum(probs, axis=(1, 2), keepdims=True)
    )

    for user_id in range(num_partitions):
        for class_idx in range(num_labels_per_partition):
            cls = (user_id + class_idx) % num_classes
            div_idx = min(user_id // safe_div, probs.shape[1] - 1)
            count = int(probs[cls, div_idx, class_idx])

            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[user_id].extend(indices)
            hist[cls] += count

    return [Subset(sorted_trainset, partition) for partition in partitions_idx]


def plot_training_history(training_history, path):
    if not training_history:
        raise ValueError("Training history must contain at least one metric.")

    num_metrics = len(training_history)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))

    if num_metrics == 1:
        axes = [axes]

    for i, (metric, values) in enumerate(training_history.items()):
        axes[i].plot(values, label=metric)
        axes[i].set_title(f"{metric.capitalize()} over Training Rounds")
        axes[i].set_xlabel("Training Round")
        axes[i].set_ylabel("Metric Value")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)

def load_merged_config(common_path="conf/config_common.yaml", local_path=None):
    common_cfg = get_cfg(common_path)
    if local_path is None:
        return dict(common_cfg)

    local_cfg = get_cfg(local_path)
    return {**common_cfg, **local_cfg}


def resolve_client_data_path(cfg, client_id=None, cli_data_path=None):
    if cli_data_path:
        return cli_data_path

    client_mode = cfg.get("client_mode", "simulation")
    base_data_path = cfg.get("data_path")

    if not base_data_path:
        raise ValueError("Missing 'data_path' in configuration.")

    if client_mode == "simulation":
        if client_id is None:
            raise ValueError("client_id is required in simulation mode.")
        return os.path.join(base_data_path, f"client{client_id}")

    if client_mode == "real":
        return base_data_path

    raise ValueError(f"Unsupported client_mode: {client_mode}")
