import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import torch
from torch import nn

from monitoring.communication import (
    bytes_to_mb,
    parameter_list_bytes,
    model_parameters_bytes,
    model_num_parameters,
    communication_summary,
)


def main():
    arrays = [
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((5,), dtype=np.float64),
    ]

    array_bytes = parameter_list_bytes(arrays)
    print("Array parameter bytes:", array_bytes)
    print("Array parameter MB:", bytes_to_mb(array_bytes))

    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )

    print("Model parameters:", model_num_parameters(model))
    print("Model bytes:", model_parameters_bytes(model))

    summary = communication_summary(model=model)
    print("Communication summary:")
    print(summary)


if __name__ == "__main__":
    main()
