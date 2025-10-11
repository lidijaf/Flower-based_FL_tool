import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
import numpy as np
import math
import os
from omegaconf import OmegaConf
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from typing import List
import matplotlib.pyplot as plt
from ruamel.yaml import YAML  # or OmegaConf if you prefer

def get_cfg(config_path=None):
    import os
    from omegaconf import OmegaConf

    base_dir = os.path.dirname(__file__)

    if config_path is None:
        # default
        config_path = os.path.join(base_dir, "conf", "config.yaml")
    else:
        # If a relative path is passed, make it relative to base_dir
        if not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    return cfg



#def get_cfg(config_path="C://Users//neman//PycharmProjects//tool_demo//conf//config.yaml"):
    """
    Load a configuration file using OmegaConf.

    Args:
        config_path (str): Path to the YAML configuration file. Default is 'conf/config.yaml'.

    Returns:
        OmegaConf object: Loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the configuration file cannot be parsed.
    """
#    if not os.path.exists(config_path):
#        raise FileNotFoundError(f"Configuration file not found at {config_path}")

#    try:
#        cfg = OmegaConf.load(config_path)
#        return cfg
#    except Exception as e:
#        raise ValueError(f"Failed to load configuration from {config_path}: {e}")


def sort_by_class(trainset: Dataset) -> Dataset:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be sorted.

    Returns
    -------
    Dataset
        The sorted training dataset.
    """
    # Ensure trainset.targets is a tensor
    if isinstance(trainset.targets, list):
        trainset.targets = torch.tensor(trainset.targets)

    # Get indices for sorting
    idxs = trainset.targets.argsort()

    # Group data by class and concatenate
    tmp = []
    tmp_targets = []

    unique_classes, class_counts = torch.unique(trainset.targets, return_counts=True)
    start = 0
    for count in class_counts:
        end = start + count
        tmp.append(Subset(trainset, idxs[start:end]))
        tmp_targets.append(trainset.targets[idxs[start:end]])
        start = end

    sorted_dataset = ConcatDataset(tmp)
    sorted_dataset.targets = torch.cat(tmp_targets)  # Concatenate targets

    return sorted_dataset


def power_law_split(sorted_trainset: Dataset) -> List[Dataset]:
    cfg = get_cfg()
    num_partitions = cfg.get("num_clients")
    num_labels_per_partition = cfg.get("num_labels_per_partition")
    min_data_per_partition = cfg.get("min_data_per_partition")
    mean = cfg.get("mean")
    sigma = cfg.get("sigma")

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
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            indices = full_idx[
                labels_cs[cls]: labels_cs[cls] + hist[cls] + min_data_per_class
            ]
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # Adjusted shape calculation for probs
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

    # Fix indexing logic
    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            # Adjusted indexing to avoid out-of-bounds error
            div_idx = min(u_id // safe_div, probs.shape[1] - 1)  # Ensure valid index
            count = int(probs[cls, div_idx, cls_idx])
            indices = full_idx[
                labels_cs[cls] + hist[cls]: labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]
    return partitions


def plot_training_history(training_history, path):
    # Determine the number of metrics and create that many subplots
    num_metrics = len(training_history)
    if num_metrics == 0:
        raise ValueError("Training history must contain at least one metric.")

    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))

    # If there's only one metric, axes will not be a list, so handle that case
    if num_metrics == 1:
        axes = [axes]

    # Plot each metric on its respective subplot
    for i, (metric, values) in enumerate(training_history.items()):
        axes[i].plot(values, label=metric)
        axes[i].set_title(f'{metric.capitalize()} over Training Rounds')
        axes[i].set_xlabel('Training Round')
        axes[i].set_ylabel('Metric Value')
        axes[i].legend()

    # Adjust layout for better visibility
    plt.tight_layout()

    # Save the plot to the specified path
    plt.savefig(path)
    plt.close(fig)


'''
MAYBE USEFUL

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
'''

