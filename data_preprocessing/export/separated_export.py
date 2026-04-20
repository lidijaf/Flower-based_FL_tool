import os
from data_preprocessing.base import PreprocessingStep


class ExportSeparatedSplitsStep(PreprocessingStep):
    def __init__(self, output_dir, include_index=False):
        self.output_dir = output_dir
        self.include_index = include_index

    def transform(self, bundle, context):
        if bundle.splits is None:
            raise ValueError("No separated splits found.")

        os.makedirs(self.output_dir, exist_ok=True)

        for client_id, split_dict in bundle.splits.items():
            safe_client_id = client_id.replace("/", "_").replace(".", "_")
            client_dir = os.path.join(self.output_dir, f"client_{safe_client_id}")
            os.makedirs(client_dir, exist_ok=True)

            for split_name, split_parts in split_dict.items():
                split_dir = os.path.join(client_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)

                if split_parts.get("X") is not None:
                    split_parts["X"].to_csv(
                        os.path.join(split_dir, "X.csv"),
                        index=self.include_index,
                    )

                if split_parts.get("y") is not None:
                    split_parts["y"].to_csv(
                        os.path.join(split_dir, "y.csv"),
                        index=self.include_index,
                    )

                if split_parts.get("meta") is not None:
                    split_parts["meta"].to_csv(
                        os.path.join(split_dir, "meta.csv"),
                        index=self.include_index,
                    )

        return bundle
