import numpy as np


def encode_parameters_for_transmission(parameters, cfg):
    precision = str(cfg.get("transmission_precision", "fp32")).lower()

    if precision == "fp16":
        return [param.astype(np.float16) for param in parameters]

    if precision == "fp32":
        return [param.astype(np.float32) for param in parameters]

    raise ValueError(f"Unsupported transmission_precision: {precision}")
