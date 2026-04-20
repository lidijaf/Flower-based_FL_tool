from data_preprocessing.base import PreprocessingStep
import pandas as pd


class AnomalyAwareSplitStep(PreprocessingStep):
    def __init__(
        self,
        label_col="is_anomaly",
        normal_value=0,
        anomaly_value=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        sort_by=None,
        reset_index=True,
        anomaly_distribution="half_half",
    ):
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-8:
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

        self.label_col = label_col
        self.normal_value = normal_value
        self.anomaly_value = anomaly_value
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.sort_by = sort_by
        self.reset_index = reset_index
        self.anomaly_distribution = anomaly_distribution

    def transform(self, bundle, context):
        if bundle.clients is None:
            raise ValueError("No client data found. Run a partitioning step before splitting.")

        client_splits = {}
        summary = {}

        for client_id, client_df in bundle.clients.items():
            df = client_df.copy()

            if self.label_col not in df.columns:
                raise ValueError(
                    f"Label column '{self.label_col}' not found in client '{client_id}'."
                )

            if self.sort_by is not None:
                missing = [col for col in self.sort_by if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing sort columns for client '{client_id}': {missing}")
                df = df.sort_values(by=self.sort_by)

            if self.reset_index:
                df = df.reset_index(drop=True)

            df_normal = df[df[self.label_col] == self.normal_value].copy()
            df_anomaly = df[df[self.label_col] == self.anomaly_value].copy()

            if self.reset_index:
                df_normal = df_normal.reset_index(drop=True)
                df_anomaly = df_anomaly.reset_index(drop=True)

            n_normal = len(df_normal)
            n_train = int(n_normal * self.train_ratio)
            n_val = int(n_normal * self.val_ratio)

            train_df = df_normal.iloc[:n_train].copy()
            val_df = df_normal.iloc[n_train:n_train + n_val].copy()
            test_df = df_normal.iloc[n_train + n_val:].copy()

            if len(df_anomaly) > 0:
                if self.anomaly_distribution == "half_half":
                    split_idx = len(df_anomaly) // 2
                    val_anomaly = df_anomaly.iloc[:split_idx].copy()
                    test_anomaly = df_anomaly.iloc[split_idx:].copy()
                elif self.anomaly_distribution == "all_test":
                    val_anomaly = df_anomaly.iloc[0:0].copy()
                    test_anomaly = df_anomaly.copy()
                else:
                    raise ValueError(
                        f"Unsupported anomaly_distribution: {self.anomaly_distribution}"
                    )

                val_df = val_df.copy()
                test_df = test_df.copy()

                val_df = val_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)

                val_df = pd.concat([val_df, val_anomaly], ignore_index=True)
                test_df = pd.concat([test_df, test_anomaly], ignore_index=True)
                
            if self.reset_index:
                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)

            client_splits[client_id] = {
                "train": train_df,
                "val": val_df,
                "test": test_df,
            }

            summary[client_id] = {
                "train_total": len(train_df),
                "train_anomalies": int((train_df[self.label_col] == self.anomaly_value).sum()),
                "val_total": len(val_df),
                "val_anomalies": int((val_df[self.label_col] == self.anomaly_value).sum()),
                "test_total": len(test_df),
                "test_anomalies": int((test_df[self.label_col] == self.anomaly_value).sum()),
            }

        bundle.splits = client_splits
        context.metadata["anomaly_split_summary"] = summary
        return bundle
