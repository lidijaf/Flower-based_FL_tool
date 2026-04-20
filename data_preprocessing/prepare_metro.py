import argparse
from data_preprocessing.recipes.metro import prepare_metro_ae_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare METRO dataset for AE FL clients")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to METRO CSV file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/metro_clients",
        help="Output directory for client data"
    )
    parser.add_argument("--num_clients", type=int, default=2, help="Number of FL clients")
    parser.add_argument(
        "--config_path",
        type=str,
        default="data_preprocessing/configs/metro_ae_schema.json",
        help="Path to preprocessing config"
    )
    args = parser.parse_args()

    prepare_metro_ae_data(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        config_path=args.config_path,
    )
