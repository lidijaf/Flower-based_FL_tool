from data_preprocessing.base import PreprocessingStep


class StringLengthFeatureStep(PreprocessingStep):
    def __init__(self, source_col, output_col):
        self.source_col = source_col
        self.output_col = output_col

    def transform(self, bundle, context):
        if self.source_col not in bundle.df.columns:
            raise ValueError(f"Column '{self.source_col}' not found in DataFrame.")

        bundle.df[self.output_col] = bundle.df[self.source_col].fillna("").astype(str).apply(len)
        return bundle
