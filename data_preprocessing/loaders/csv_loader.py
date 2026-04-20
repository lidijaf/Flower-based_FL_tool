import pandas as pd
from data_preprocessing.base import DatasetBundle, PreprocessingStep


class CSVLoaderStep(PreprocessingStep):
    def __init__(self, path):
        self.path = path

    def transform(self, bundle, context):
        df = pd.read_csv(self.path)
        return DatasetBundle(df=df)
