import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing.base import PreprocessingContext
from data_preprocessing.recipes.act import build_act_standard_pipeline
from data_preprocessing.export.tensor_export import ExportTensorSplitsStep


context = PreprocessingContext()

pipeline = build_act_standard_pipeline(
    input_path="data/raw/run1.json",
    output_dir="data/processed/act_standard_csv",
)

# Replace the last CSV export step with tensor export
pipeline.steps[-1] = ExportTensorSplitsStep(
    output_dir="data/processed/act_standard_pt"
)

bundle = pipeline.fit_transform(None, context)

print("Split clients:", list(bundle.splits.keys())[:5])
print("Number of clients:", len(bundle.splits))

first_client = list(bundle.splits.keys())[0]
print("First client:", first_client)
print("Train shape:", bundle.splits[first_client]["train"].shape)
print("Val shape:", bundle.splits[first_client]["val"].shape)
print("Test shape:", bundle.splits[first_client]["test"].shape)
