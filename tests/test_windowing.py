import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data_preprocessing.transforms.windowing import create_sliding_windows


def main():
    data = torch.arange(20, dtype=torch.float32).view(10, 2)

    windows = create_sliding_windows(data, win_size=4, step=2)

    print("Input shape:", data.shape)
    print("Windows shape:", windows.shape)
    print("First window:")
    print(windows[0])


if __name__ == "__main__":
    main()
