import argparse
from data_preprocessing.prepare_dataset import prepare_dataset


BUILTIN_DATASETS = {"mnist", "cifar10", "fmnist"}


def main():
    parser = argparse.ArgumentParser(
        description="Download and/or prepare a dataset for federated learning experiments."
    )

    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name, e.g. mnist, cifar10, fmnist, act, metro_ae, metropt, psm, tabular",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to input file or directory for local/custom datasets.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for prepared data. If omitted, a default path will be used.",
    )

    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Number of clients to create. If omitted, defaults to 1.",
    )

    parser.add_argument(
        "--split",
        type=int,
        default=None,
        help="Alias for --num_clients. If omitted, defaults to 1.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "balanced", "anomaly"],
        help="Preparation mode. Mostly relevant for datasets like ACT.",
    )

    args = parser.parse_args()

    if args.num_clients is not None and args.split is not None:
        if args.num_clients != args.split:
            raise ValueError(
                f"--num_clients ({args.num_clients}) and --split ({args.split}) "
                f"were both provided but do not match."
            )
        resolved_num_clients = args.num_clients
    elif args.num_clients is not None:
        resolved_num_clients = args.num_clients
    elif args.split is not None:
        resolved_num_clients = args.split
    else:
        resolved_num_clients = 1

    if resolved_num_clients < 1:
        raise ValueError("Number of clients must be at least 1.")

    if args.output_dir is None:
        if args.dataset_name in BUILTIN_DATASETS:
            args.output_dir = f"data/{args.dataset_name}"
        else:
            args.output_dir = f"data/processed/{args.dataset_name}_{args.mode}"

    prepare_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        input_path=args.input_path,
        num_clients=resolved_num_clients,
        mode=args.mode,
    )

    print("\nDataset preparation completed successfully.")
    print(f"Dataset: {args.dataset_name}")
    print(f"Mode: {args.mode}")
    print(f"Clients: {resolved_num_clients}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
