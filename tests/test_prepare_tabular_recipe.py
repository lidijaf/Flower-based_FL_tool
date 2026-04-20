import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.recipes.tabular import prepare_tabular_fl_data

prepare_tabular_fl_data(
    input_csv="data/raw/sample_tabular.csv",
    output_dir="data/processed/tabular_clients",
    num_clients=3,
    val_ratio=0.2,
    test_ratio=0.2,
    label_col=0,
)
