import torch


def create_sliding_windows(data, win_size: int, step: int = 1):
    """
    Convert flat time-series data [N, features] into windows
    [num_windows, win_size, features].
    """
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)

    if data.dim() != 2:
        raise ValueError(
            f"Expected 2D input [num_samples, num_features], got shape {tuple(data.shape)}"
        )

    if win_size <= 0:
        raise ValueError("win_size must be positive")

    if step <= 0:
        raise ValueError("step must be positive")

    n_samples = data.shape[0]

    if n_samples < win_size:
        raise ValueError(
            f"Cannot create windows: num_samples={n_samples} < win_size={win_size}"
        )

    windows = []

    for start in range(0, n_samples - win_size + 1, step):
        end = start + win_size
        windows.append(data[start:end])

    return torch.stack(windows)
