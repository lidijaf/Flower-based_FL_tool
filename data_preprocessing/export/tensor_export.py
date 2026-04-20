import os
import torch
import pandas as pd
from data_preprocessing.base import PreprocessingStep


class ExportTensorSplitsStep(PreprocessingStep):
    def __init__(self, output_dir, dtype=torch.float32, exclude_cols=None):
        self.output_dir = output_dir
        self.dtype = dtype
        self.exclude_cols = exclude_cols or []

    def transform(self, bundle, context):
        if bundle.splits is None:
            raise ValueError("No splits found. Run splitting before tensor export.")

        os.makedirs(self.output_dir, exist_ok=True)

        group_mapping = (
            context.metadata.get("group_partition", {}).get("client_mapping", {})
        )
        balanced_mapping = (
            context.metadata.get("balanced_partition", {}).get("client_groups", {})
        )

        mapping_rows = []

        for client_id, splits in bundle.splits.items():
            client_dir = os.path.join(self.output_dir, client_id)
            os.makedirs(client_dir, exist_ok=True)

            if client_id in group_mapping:
                original_client_id = group_mapping[client_id]
            elif client_id in balanced_mapping:
                original_client_id = ",".join(str(x) for x in balanced_mapping[client_id])
            else:
                original_client_id = client_id

            mapping_rows.append({
                "exported_client_name": client_id,
                "original_client_id": original_client_id,
            })

            for split_name, df in splits.items():
                numeric_df = df.select_dtypes(include=["number"]).copy()

                cols_to_drop = [col for col in self.exclude_cols if col in numeric_df.columns]
                if cols_to_drop:
                    numeric_df = numeric_df.drop(columns=cols_to_drop)

                tensor = torch.tensor(numeric_df.values, dtype=self.dtype)
                torch.save(tensor, os.path.join(client_dir, f"{split_name}.pt"))

        mapping_df = pd.DataFrame(mapping_rows)
        mapping_df.to_csv(
            os.path.join(self.output_dir, "client_mapping.csv"),
            index=False
        )

        return bundle
