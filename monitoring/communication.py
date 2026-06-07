import numpy as np
import torch


def bytes_to_mb(num_bytes):
    return float(num_bytes) / (1024 * 1024)


def numpy_array_bytes(array):
    array = np.asarray(array)
    return int(array.nbytes)


def torch_tensor_bytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def parameter_list_bytes(parameters):
    total = 0

    for param in parameters:
        if isinstance(param, np.ndarray):
            total += numpy_array_bytes(param)
        elif torch.is_tensor(param):
            total += torch_tensor_bytes(param)
        else:
            total += np.asarray(param).nbytes

    return int(total)


def model_parameters_bytes(model):
    total = 0

    for param in model.parameters():
        total += torch_tensor_bytes(param.data)

    return int(total)


def model_num_parameters(model, trainable_only=False):
    params = model.parameters()

    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))

    return int(sum(p.numel() for p in params))


def communication_summary(parameters=None, model=None):
    if parameters is None and model is None:
        raise ValueError("Either parameters or model must be provided.")

    if parameters is not None:
        total_bytes = parameter_list_bytes(parameters)
        num_parameters = int(sum(np.asarray(p).size for p in parameters))
    else:
        total_bytes = model_parameters_bytes(model)
        num_parameters = model_num_parameters(model)

    return {
        "num_parameters": num_parameters,
        "total_bytes": total_bytes,
        "total_mb": bytes_to_mb(total_bytes),
    }
