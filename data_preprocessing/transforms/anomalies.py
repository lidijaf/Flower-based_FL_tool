import numpy as np
import pandas as pd
from data_preprocessing.base import PreprocessingStep


class InjectSyntheticAnomaliesStep(PreprocessingStep):
    def __init__(
        self,
        anomaly_fraction=0.05,
        numeric_cols=None,
        categorical_cols=None,
        temporal_cols=None,
        label_col="is_anomaly",
        noise_std=0.5,
        shift_multiplier=3.0,
        random_state=42,
    ):
        self.anomaly_fraction = anomaly_fraction
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.temporal_cols = temporal_cols or []
        self.label_col = label_col
        self.noise_std = noise_std
        self.shift_multiplier = shift_multiplier
        self.random_state = random_state

    def transform(self, bundle, context):
        df = bundle.df.copy()

        rng = np.random.default_rng(self.random_state)

        if self.label_col not in df.columns:
            df[self.label_col] = 0

        n_rows = len(df)
        n_anomalies = int(n_rows * self.anomaly_fraction)

        if n_anomalies == 0:
            bundle.df = df
            return bundle

        anomaly_indices = rng.choice(df.index.to_numpy(), size=n_anomalies, replace=False)

        # numeric perturbation
        for col in self.numeric_cols:
            if col not in df.columns:
                continue

            #FIX: ensure float dtype before adding noise
            df[col] = df[col].astype(float)

            col_std = df[col].std()
            if pd.isna(col_std) or col_std == 0:
                col_std = 1.0

            noise = rng.normal(loc=0.0, scale=self.noise_std, size=n_anomalies)
            shift = self.shift_multiplier * col_std

            df.loc[anomaly_indices, col] = df.loc[anomaly_indices, col] + noise + shift
        
        # temporal perturbation
        for col in self.temporal_cols:
            if col not in df.columns:
                continue
            col_std = df[col].std()
            if pd.isna(col_std) or col_std == 0:
                col_std = 1.0

            shift = self.shift_multiplier * col_std
            df.loc[anomaly_indices, col] = df.loc[anomaly_indices, col] + shift

        # categorical perturbation
        for col in self.categorical_cols:
            if col not in df.columns:
                continue

            unique_values = df[col].dropna().astype(str).unique().tolist()
            if len(unique_values) < 2:
                continue

            current_values = df.loc[anomaly_indices, col].astype(str)
            replacement_values = []

            for current in current_values:
                candidates = [v for v in unique_values if v != current]
                replacement_values.append(rng.choice(candidates))

            df.loc[anomaly_indices, col] = replacement_values

        df.loc[anomaly_indices, self.label_col] = 1

        context.metadata["anomaly_injection"] = {
            "label_col": self.label_col,
            "anomaly_fraction": self.anomaly_fraction,
            "n_rows": n_rows,
            "n_anomalies": n_anomalies,
            "random_state": self.random_state,
        }

        bundle.df = df
        return bundle
