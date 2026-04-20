import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.prepare_dataset import prepare_dataset


def test_mnist():
    print("\n=== TEST: MNIST ===")
    prepare_dataset(
        dataset_name="mnist",
        output_dir="data/raw/mnist_test"
    )


def test_act_standard():
    print("\n=== TEST: ACT (standard) ===")
    prepare_dataset(
        dataset_name="act",
        input_path="data/raw/run1.json",
        output_dir="data/processed/act_standard_test",
        mode="standard",
    )


def test_act_balanced():
    print("\n=== TEST: ACT (balanced) ===")
    prepare_dataset(
        dataset_name="act",
        input_path="data/raw/run1.json",
        output_dir="data/processed/act_balanced_test",
        mode="balanced",
        num_clients=4,
    )


def test_act_anomaly():
    print("\n=== TEST: ACT (anomaly) ===")
    prepare_dataset(
        dataset_name="act",
        input_path="data/raw/run1.json",
        output_dir="data/processed/act_anomaly_test",
        mode="anomaly",
    )


def test_metro():
    print("\n=== TEST: METRO AE ===")
    prepare_dataset(
        dataset_name="metro_ae",
        input_path="data/raw/metro.csv",
        output_dir="data/processed/metro_test",
        num_clients=2,
    )


def test_tabular():
    print("\n=== TEST: TABULAR ===")
    prepare_dataset(
        dataset_name="tabular",
        input_path="data/raw/sample_tabular.csv",
        output_dir="data/processed/tabular_test",
        num_clients=3,
    )


if __name__ == "__main__":
    # Run only what you want
    test_mnist()
    test_act_standard()
    test_act_balanced()
    test_act_anomaly()
    test_tabular()
    # test_metro()  # enable if metro.csv exists
