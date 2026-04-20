from sklearn.preprocessing import StandardScaler
from data_preprocessing.base import PreprocessingStep


class StandardScaleStep(PreprocessingStep):
    def __init__(self, columns, artifact_name="scaler", copy=True):
        self.columns = columns
        self.artifact_name = artifact_name
        self.copy = copy
        self.scaler = StandardScaler()

    def fit(self, bundle, context):
        df = bundle.df

        missing = [col for col in self.columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found for scaling: {missing}")

        self.scaler.fit(df[self.columns])
        context.artifacts[self.artifact_name] = self.scaler
        return self

    def transform(self, bundle, context):
        df = bundle.df.copy() if self.copy else bundle.df

        missing = [col for col in self.columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found for scaling: {missing}")

        df[self.columns] = self.scaler.transform(df[self.columns])
        bundle.df = df
        return bundle
