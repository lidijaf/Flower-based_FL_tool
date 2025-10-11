import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from utils import get_cfg
from ruamel.yaml import YAML
# import os
from sklearn.preprocessing import StandardScaler

import os
base_dir = os.path.dirname(os.path.dirname(__file__))
#config_file_path = os.path.join(base_dir, "conf", "config.yaml")
data_path = os.path.join(base_dir, "data/Pooled Server Metrics (PSM)/client2")

from omegaconf import OmegaConf

shared_cfg = OmegaConf.to_container(get_cfg("conf/config_common.yaml"), resolve=True)
client_cfg = OmegaConf.to_container(get_cfg("conf/config_client.yaml"), resolve=True)

# merge
config_file_path = {**shared_cfg, **client_cfg}


#config_file_path = "/conf/config.yaml"
yaml = YAML()

# Global variable to store the dropped feature mask
dropped_feature_mask = None


def set_dropped_feature_mask(total_features: int, drop_ratio: float = 0, seed: int = 42):
    """Generate a fixed mask for dropping a consistent set of features."""
    global dropped_feature_mask
    np.random.seed(seed)
    mask = np.ones(total_features, dtype=bool)
    num_drop = int(total_features * drop_ratio)
    drop_indices = np.random.choice(total_features, num_drop, replace=False)
    mask[drop_indices] = False
    dropped_feature_mask = mask
    print(f"Dropping {num_drop} out of {total_features} features. Dropped indices: {drop_indices}")

    # Update config.yaml
    with open(config_file_path, "r") as file:
        config = yaml.load(file)
    config['input_c'] = total_features - num_drop
    config['output_c'] = total_features - num_drop
    with open(config_file_path, "w") as file:
        yaml.dump(config, file)


def apply_feature_mask(data: np.ndarray) -> np.ndarray:
    """Apply the globally set feature mask to the given data."""
    global dropped_feature_mask
    if dropped_feature_mask is None:
        raise RuntimeError("Feature mask not initialized. Call set_dropped_feature_mask() first.")
    return data[:, dropped_feature_mask]


class PSMDataset(Dataset):
    def __init__(self, data, labels, cfg, mode="train"):
        #cfg = get_cfg()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.win_size = cfg.get("win_size")
        self.step = cfg.get("step")
        self.mode = mode

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.data.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(self.labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.data[index:index + self.win_size]), np.float32(
                self.labels[index:index + self.win_size])
        else:
            return np.float32(self.data[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


def load_data(cfg, mislabel_percent: float = 0.0) -> Tuple[DataLoader, DataLoader]:
    assert 0 <= mislabel_percent <= 1, "mislabel_percent must be between 0 and 1"

    # PSM data
    #cfg = get_cfg()
    #data_path = cfg["data_path"]
    # data_path = 'C://Users//neman//PycharmProjects//data//Pooled Server Metrics (PSM) split//client1'
    scaler = StandardScaler()
    data = pd.read_csv(data_path + '/train.csv')
    data = data.values[:, 1:]

    data = np.nan_to_num(data)

    scaler.fit(data)
    data = scaler.transform(data)
    test_data = pd.read_csv(data_path + '/test.csv')

    test_data = test_data.values[:, 1:]
    test_data = np.nan_to_num(test_data)

    test = scaler.transform(test_data)

    '''
        # Set feature drop mask once
        if dropped_feature_mask is None:
            set_dropped_feature_mask(total_features=normal_data.shape[1] - 1, drop_ratio=0, seed=cfg.get("seed"))
            set_dropped_feature_mask(total_features=data.shape[1] - 2, drop_ratio=0, seed=cfg.get("seed"))

        # Apply mask to input features (exclude label column)
        #normal_data_features = apply_feature_mask(normal_data[:, :-1])
        #anomaly_data_features = apply_feature_mask(anomaly_data[:, :-1])
        #normal_labels = normal_data[:, -1]
        #anomaly_labels = anomaly_data[:, -1]
        '''

    train = data
    val = test

    test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

    print("test:", test.shape)
    print("train:", train.shape)

    train_dataset = PSMDataset(train, test_labels, cfg, mode="train")
    val_dataset = PSMDataset(val, test_labels, cfg, mode="val")
    test_dataset = PSMDataset(test, test_labels, cfg, mode="test")

    batch_size = cfg.get("batch_size")
    num_workers = cfg.get("num_workers")

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return trainloader, testloader
