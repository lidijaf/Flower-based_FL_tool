import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.prepare_dataset import prepare_builtin_dataset

prepare_builtin_dataset(
    dataset_name="mnist",
    data_dir="data/raw/mnist"
)
