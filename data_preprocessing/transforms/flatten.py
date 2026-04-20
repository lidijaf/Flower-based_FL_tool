import pandas as pd
from data_preprocessing.base import PreprocessingStep


class FlattenJSONStep(PreprocessingStep):
    def __init__(self, flatten_fn):
        self.flatten_fn = flatten_fn

    def transform(self, bundle, context):
        records = bundle.df.to_dict(orient="records")
        flat_data = [self.flatten_fn(record) for record in records]
        bundle.df = pd.DataFrame(flat_data)
        return bundle
