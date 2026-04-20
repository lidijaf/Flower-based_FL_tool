import os
import pandas as pd
from data_preprocessing.base import PreprocessingStep


class ExportCSVSplitsStep(PreprocessingStep):
    def __init__(self, output_dir, include_index=False):
        self.output_dir = output_dir
        self.include_index = include_index

    def transform(self, bundle, context):
        if bundle.splits is None:
            raise ValueError("No splits found. Run splitting before export.")

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

            original_client_id = None

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
                path = os.path.join(client_dir, f"{split_name}.csv")
                df.to_csv(path, index=self.include_index)

        mapping_df = pd.DataFrame(mapping_rows)
        mapping_df.to_csv(
            os.path.join(self.output_dir, "client_mapping.csv"),
            index=False
        )

        return bundle
