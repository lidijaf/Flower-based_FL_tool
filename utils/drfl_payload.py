import io
import numpy as np


def extract_flat_gradients(model) -> np.ndarray:
    """
    Extract gradients from a trained model and flatten them into one 1D vector.
    """
    flat_grads = []

    for param in model.parameters():
        if param.grad is None:
            flat_grads.append(np.zeros(param.numel(), dtype=np.float32))
        else:
            flat_grads.append(
                param.grad.detach().cpu().numpy().astype(np.float32).ravel()
            )

    if not flat_grads:
        return np.array([], dtype=np.float32)

    return np.concatenate(flat_grads)


def serialize_gradient_vector(grad_vector: np.ndarray) -> bytes:
    """
    Serialize a numpy gradient vector into bytes for Flower metrics transport.
    """
    buffer = io.BytesIO()
    np.save(buffer, grad_vector.astype(np.float32), allow_pickle=False)
    return buffer.getvalue()


def deserialize_gradient_vector(blob: bytes) -> np.ndarray:
    """
    Deserialize gradient bytes back into a numpy vector.
    """
    buffer = io.BytesIO(blob)
    return np.load(buffer, allow_pickle=False)


def gradients_to_metrics(grad_vector: np.ndarray) -> dict:
    """
    Convert gradient vector to Flower-metrics-safe representation.
    """
    return {
        "gradients_blob": serialize_gradient_vector(grad_vector),
        "gradient_dim": int(grad_vector.shape[0]),
    }
