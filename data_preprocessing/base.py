from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd


@dataclass
class PreprocessingContext:
    params: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    X: Optional[pd.DataFrame] = None
    y: Optional[pd.Series] = None
    splits: Optional[Dict[str, Any]] = None
    clients: Optional[Dict[str, pd.DataFrame]] = None


class PreprocessingStep:
    def fit(self, bundle: DatasetBundle, context: PreprocessingContext):
        return self

    def transform(self, bundle: DatasetBundle, context: PreprocessingContext) -> DatasetBundle:
        return bundle

    def fit_transform(self, bundle: DatasetBundle, context: PreprocessingContext) -> DatasetBundle:
        self.fit(bundle, context)
        return self.transform(bundle, context)
