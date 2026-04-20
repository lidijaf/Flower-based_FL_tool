from data_preprocessing.base import PreprocessingStep


class DeltaTimeFeatureStep(PreprocessingStep):
    def __init__(self, timestamp_col, output_col="delta_t", group_by=None, scale_factor=1.0, fillna_value=0):
        self.timestamp_col = timestamp_col
        self.output_col = output_col
        self.group_by = group_by
        self.scale_factor = scale_factor
        self.fillna_value = fillna_value

    def transform(self, bundle, context):
        df = bundle.df

        if self.timestamp_col not in df.columns:
            raise ValueError(f"Column '{self.timestamp_col}' not found in DataFrame.")

        if self.group_by is not None:
            if self.group_by not in df.columns:
                raise ValueError(f"Group column '{self.group_by}' not found in DataFrame.")
            delta = df.groupby(self.group_by)[self.timestamp_col].diff()
        else:
            delta = df[self.timestamp_col].diff()

        bundle.df[self.output_col] = delta.fillna(self.fillna_value) / self.scale_factor
        return bundle
