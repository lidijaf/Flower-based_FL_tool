import argparse
from data_preprocessing.recipes.tabular import prepare_tabular_fl_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split tabular dataset for FL clients")

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data/clients")
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--label_col", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    label_col = None if args.label_col < 0 else args.label_col

    prepare_tabular_fl_data(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        label_col=label_col,
        random_seed=args.seed,
    )
