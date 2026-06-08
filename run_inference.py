import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from utils import get_cfg
from utils.model_parameters import build_model_from_cfg, load_model_state
from evaluation.results import save_predictions_csv, save_json


def load_checkpoint_metadata(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        return {}

    return {
        "model": checkpoint.get("model"),
        "dataset": checkpoint.get("dataset"),
        "algorithm": checkpoint.get("algorithm"),
    }

def load_config():
    common_cfg = get_cfg("conf/config_common.yaml")
    return dict(common_cfg)


def load_input_data(input_path):
    data = torch.load(input_path)

    if isinstance(data, TensorDataset):
        x = data.tensors[0]
        y = data.tensors[1] if len(data.tensors) > 1 else None
        return x, y

    if isinstance(data, Dataset):
        return data, None

    if isinstance(data, tuple):
        x = data[0]
        y = data[1] if len(data) > 1 else None
        return x, y

    if torch.is_tensor(data):
        return data, None

    raise TypeError(f"Unsupported input data type: {type(data)}")

def run_model_inference(model, data, cfg):
    device = torch.device(cfg.get("device") or "cpu")
    batch_size = cfg.get("batch_size", 32)

    model.to(device)
    model.eval()

    if isinstance(data, Dataset):
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)

    outputs = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                batch_x = batch[0]
                batch_y = batch[1] if len(batch) > 1 else None
            else:
                batch_x = batch
                batch_y = None

            batch_x = batch_x.to(device)
            output = model(batch_x)

            if isinstance(output, tuple):
                output = output[0]

            outputs.append(output.detach().cpu())

            if batch_y is not None:
                labels.append(batch_y.detach().cpu())

    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0) if labels else None

    return outputs, labels

def classification_predictions(outputs):
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    return predictions.numpy(), probabilities.numpy()


def reconstruction_errors(outputs, x):
    errors = torch.mean((outputs.cpu() - x.cpu()) ** 2, dim=tuple(range(1, outputs.dim())))
    return errors.numpy()


def main():
    parser = argparse.ArgumentParser(description="Run inference using a saved global model.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", default="outputs/inference")
    parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    cfg = load_config()

    metadata = load_checkpoint_metadata(args.model_path)
    for key, value in metadata.items():
        if value is not None:
            cfg[key] = value

    model = build_model_from_cfg(cfg)
    model = load_model_state(model, args.model_path, device=cfg.get("device") or "cpu")
    
    data, y = load_input_data(args.input_path)
    outputs, inferred_y = run_model_inference(model, data, cfg)

    if y is None:
        y = inferred_y
    
    task = cfg.get("task")
    model_name = cfg.get("model")

    os.makedirs(args.output_dir, exist_ok=True)

    if task == "classification":
        predictions, probabilities = classification_predictions(outputs)

        np.save(os.path.join(args.output_dir, "predictions.npy"), predictions)
        np.save(os.path.join(args.output_dir, "probabilities.npy"), probabilities)

        report = {
            "task": task,
            "model": model_name,
            "num_samples": int(len(predictions)),
            "prediction_shape": list(predictions.shape),
            "probability_shape": list(probabilities.shape),
        }

    elif task == "anomaly detection":
        if isinstance(data, Dataset):
            raise ValueError("Reconstruction-error inference requires tensor input data, not a torchvision Dataset.")

        errors = reconstruction_errors(outputs, data)

        np.save(os.path.join(args.output_dir, "scores.npy"), errors)

        report = {
            "task": task,
            "model": model_name,
            "num_samples": int(len(errors)),
            "score_shape": list(errors.shape),
        }

        if args.threshold is not None:
            predictions = (errors > args.threshold).astype(int)
            np.save(os.path.join(args.output_dir, "predictions.npy"), predictions)
            report["threshold"] = float(args.threshold)
            report["prediction_shape"] = list(predictions.shape)

    else:
        raise ValueError(f"Unsupported task: {task}")

    if y is not None:
        np.save(os.path.join(args.output_dir, "labels.npy"), y.detach().cpu().numpy())
        report["label_shape"] = list(y.shape)

    save_json(report, os.path.join(args.output_dir, "inference_report.json"))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
