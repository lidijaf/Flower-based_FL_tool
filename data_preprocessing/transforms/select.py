from data_preprocessing.base import PreprocessingStep


class SeparateFeaturesAndLabelsStep(PreprocessingStep):
    def __init__(
        self,
        label_cols=None,
        drop_cols=None,
        feature_cols=None,
        metadata_cols=None,
    ):
        self.label_cols = label_cols or []
        self.drop_cols = drop_cols or []
        self.feature_cols = feature_cols
        self.metadata_cols = metadata_cols or []

    def transform(self, bundle, context):
        if bundle.splits is None:
            raise ValueError("No splits found. Run a splitting step before separating features and labels.")

        separated = {}

        for client_id, split_dict in bundle.splits.items():
            separated[client_id] = {}

            for split_name, df in split_dict.items():
                df = df.copy()

                missing_labels = [col for col in self.label_cols if col not in df.columns]
                if missing_labels:
                    raise ValueError(
                        f"Missing label columns in client '{client_id}', split '{split_name}': {missing_labels}"
                    )

                missing_meta = [col for col in self.metadata_cols if col not in df.columns]
                if missing_meta:
                    raise ValueError(
                        f"Missing metadata columns in client '{client_id}', split '{split_name}': {missing_meta}"
                    )

                y = df[self.label_cols].copy() if self.label_cols else None
                meta = df[self.metadata_cols].copy() if self.metadata_cols else None

                if self.feature_cols is not None:
                    missing_features = [col for col in self.feature_cols if col not in df.columns]
                    if missing_features:
                        raise ValueError(
                            f"Missing feature columns in client '{client_id}', split '{split_name}': {missing_features}"
                        )
                    X = df[self.feature_cols].copy()
                else:
                    excluded = set(self.label_cols + self.drop_cols + self.metadata_cols)
                    X = df.drop(columns=[col for col in excluded if col in df.columns]).copy()

                separated[client_id][split_name] = {
                    "X": X,
                    "y": y,
                    "meta": meta,
                }

        context.metadata["separated_features_labels"] = True
        bundle.splits = separated
        return bundle
