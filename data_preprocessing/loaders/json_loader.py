import json
import pandas as pd
from data_preprocessing.base import DatasetBundle, PreprocessingStep


class JSONLoaderStep(PreprocessingStep):
    def __init__(self, path):
        self.path = path

    def transform(self, bundle, context):
        with open(self.path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return DatasetBundle(df=df)
