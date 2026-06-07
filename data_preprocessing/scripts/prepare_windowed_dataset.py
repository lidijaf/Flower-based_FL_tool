import argparse
import json
import os

from data_preprocessing.export.windowed_tensor_export import window_all_clients


def main():
    parser = argparse.ArgumentParser(
        description="Create windowed time-series tensor datasets for Transformer models."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--win_size", type=int, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--label_mode",
        choices=["sequence", "last", "max"],
        default="sequence",
    )

    args = parser.parse_args()

    summaries = window_all_clients(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        win_size=args.win_size,
        step=args.step,
        label_mode=args.label_mode,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    summary_path = os.path.join(args.output_dir, "windowing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Windowed dataset written to: {args.output_dir}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
