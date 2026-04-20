from data_preprocessing.datasets.registry import DATASET_REGISTRY as VISION_DATASETS
from data_preprocessing.recipes.metro import prepare_metro_ae_data
from data_preprocessing.recipes.tabular import prepare_tabular_fl_data
from data_preprocessing.recipes.psm import prepare_psm_data
from data_preprocessing.recipes.act import (
    build_act_standard_pipeline,
    build_act_balanced_pipeline,
    build_act_anomaly_pipeline,
)


UNIFIED_DATASET_REGISTRY = {
    "mnist": {
        "type": "vision",
        "download": VISION_DATASETS["mnist"],
    },
    "cifar10": {
        "type": "vision",
        "download": VISION_DATASETS["cifar10"],
    },
    "fmnist": {
        "type": "vision",
        "download": VISION_DATASETS["fmnist"],
    },
    "metropt": {
        "type": "tabular_anomaly",
        "prepare": prepare_metro_ae_data,
    },
    "psm": {
        "type": "timeseries",
        "prepare": prepare_psm_data,
    },
    "tabular": {
        "type": "tabular",
        "prepare": prepare_tabular_fl_data,
    },
    "act": {
        "type": "structured",
        "prepare_pipeline": build_act_standard_pipeline,
        "prepare_balanced_pipeline": build_act_balanced_pipeline,
        "prepare_anomaly_pipeline": build_act_anomaly_pipeline,
    },
}
