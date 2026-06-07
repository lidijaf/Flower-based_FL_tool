import os
import torch

from data_preprocessing.transforms.windowing import create_sliding_windows


def create_windowed_labels(labels, win_size: int, step: int = 1, mode: str = "sequence"):
    """
    Convert labels [N] into window labels.

    mode:
      - "sequence": return labels for every time point in each window,
                    shape [num_windows, win_size]
      - "last": label of the last point in each window,
                shape [num_windows]
      - "max": window is anomalous if any point is anomalous,
               shape [num_windows]
    """
    if labels is None:
        return None

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    if labels.dim() != 1:
        labels = labels.view(-1)

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    if labels.dim() != 1:
        labels = labels.view(-1)

    # Convert MetroPT labels to binary anomaly labels
    labels = (labels > 0).long()

    window_labels = []

    for start in range(0, len(labels) - win_size + 1, step):
        end = start + win_size
        window = labels[start:end]

        if mode == "sequence":
            window_labels.append(window)
        elif mode == "last":
            window_labels.append(window[-1])
        elif mode == "max":
            window_labels.append(window.max())
        else:
            raise ValueError("mode must be 'sequence', 'last', or 'max'")

    return torch.stack(window_labels)

def window_split_file(input_path, output_path, win_size: int, step: int = 1, label_mode: str = "sequence"):
    data = torch.load(input_path)

    if isinstance(data, tuple):
        x = data[0]
        y = data[1] if len(data) > 1 else None
    else:
        x = data
        y = None

    if x.shape[0] < win_size:
        return {
            "input_path": input_path,
            "output_path": output_path,
            "input_shape": tuple(x.shape),
            "output_shape": None,
            "has_labels": y is not None,
            "label_shape": None,
            "skipped": True,
            "reason": f"num_samples={x.shape[0]} < win_size={win_size}",
        }

    x_windows = create_sliding_windows(x, win_size=win_size, step=step)
    y_windows = create_windowed_labels(y, win_size=win_size, step=step, mode=label_mode)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if y_windows is None:
        torch.save(x_windows, output_path)
    else:
        torch.save((x_windows, y_windows), output_path)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "input_shape": tuple(x.shape),
        "output_shape": tuple(x_windows.shape),
        "has_labels": y_windows is not None,
        "label_shape": None if y_windows is None else tuple(y_windows.shape),
    }


def window_client_dataset(input_client_dir, output_client_dir, win_size: int, step: int = 1, label_mode: str = "sequence"):
    summaries = {}

    for split in ["train", "val", "test"]:
        input_path = os.path.join(input_client_dir, f"{split}.pt")

        if not os.path.exists(input_path):
            continue

        output_path = os.path.join(output_client_dir, f"{split}.pt")

        summaries[split] = window_split_file(
            input_path=input_path,
            output_path=output_path,
            win_size=win_size,
            step=step,
            label_mode=label_mode,
        )

    return summaries


def window_all_clients(input_dir, output_dir, win_size: int, step: int = 1, label_mode: str = "sequence"):
    all_summaries = {}

    for name in sorted(os.listdir(input_dir)):
        input_client_dir = os.path.join(input_dir, name)

        if not os.path.isdir(input_client_dir):
            continue

        output_client_dir = os.path.join(output_dir, name)

        all_summaries[name] = window_client_dataset(
            input_client_dir=input_client_dir,
            output_client_dir=output_client_dir,
            win_size=win_size,
            step=step,
            label_mode=label_mode,
        )

    return all_summaries
