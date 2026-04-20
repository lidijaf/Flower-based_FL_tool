import pandas as pd
from data_preprocessing.base import PreprocessingStep


class OneHotEncodeStep(PreprocessingStep):
    def __init__(self, columns, drop_first=False, dummy_na=False, fill_missing=True, fill_value="NA"):
        self.columns = columns
        self.drop_first = drop_first
        self.dummy_na = dummy_na
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, bundle, context):
        df = bundle.df.copy()

        missing = [col for col in self.columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found for encoding: {missing}")

        if self.fill_missing:
            for col in self.columns:
                df[col] = df[col].fillna(self.fill_value).astype(str)

        df = pd.get_dummies(
            df,
            columns=self.columns,
            drop_first=self.drop_first,
            dummy_na=self.dummy_na,
        )

        bundle.df = df
        return bundle
