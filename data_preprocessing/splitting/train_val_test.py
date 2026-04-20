from data_preprocessing.base import PreprocessingStep


class TrainValTestSplitStep(PreprocessingStep):
    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        sort_by=None,
        reset_index=True,
    ):
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-8:
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.sort_by = sort_by
        self.reset_index = reset_index

    def transform(self, bundle, context):
        if bundle.clients is None:
            raise ValueError("No client data found. Run a partitioning step before splitting.")

        client_splits = {}

        for client_id, client_df in bundle.clients.items():
            df = client_df.copy()

            if self.sort_by is not None:
                missing = [col for col in self.sort_by if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing sort columns for client '{client_id}': {missing}")
                df = df.sort_values(by=self.sort_by)

            if self.reset_index:
                df = df.reset_index(drop=True)

            n = len(df)
            n_train = int(n * self.train_ratio)
            n_val = int(n * (self.train_ratio + self.val_ratio))

            client_splits[client_id] = {
                "train": df.iloc[:n_train].reset_index(drop=True),
                "val": df.iloc[n_train:n_val].reset_index(drop=True),
                "test": df.iloc[n_val:].reset_index(drop=True),
            }

        bundle.splits = client_splits
        return bundle
